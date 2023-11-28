from collections import OrderedDict
from typing import Tuple, Dict, Optional, List, Sequence
from typing import TypeVar

import os
from datetime import datetime

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.embodiedai.models.fusion_models import Fusion
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger

FusionType = TypeVar("FusionType", bound=Fusion)


class VisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    action_space: gym.spaces.Discrete

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        auxiliary_model_class=AuxiliaryModel,
    ):
        # print("init visualNavAC")
        super().__init__(action_space=action_space, observation_space=observation_space)
        # print("init visualNavAC")
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        # Define the placeholders in init function
        self.state_encoders: Optional[nn.ModuleDict] = None
        self.aux_models: Optional[nn.ModuleDict] = None
        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None
        self.prev_action_embedder: Optional[FeatureEmbedding] = None

        self.fusion_model: Optional[nn.Module] = None
        self.belief_names: Optional[Sequence[str]] = None
        self.auxiliary_model_class = auxiliary_model_class
        # self.coordinate_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, 128),  # First hidden layer
        #     nn.ReLU(),                    # Activation function
        #     nn.Linear(128, 64),           # Second hidden layer
        #     nn.ReLU(),                    # Activation function
        #     nn.Linear(64, 2)              # Output layer for 2D coordinates
        # )
        # self.mlp_loss_function = nn.MSELoss()  # Mean Squared Error Loss
        # self.mlp_optimizer = optim.Adam(self.coordinate_mlp.parameters(), lr=0.001)
        self.create_storage_directory()
        self.data_storage = []
        # self.coordinate_mlp.apply(self.init_weights)

    # # Initialize the weights of coordinate_mlp
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)

    def create_storage_directory(self):
        # Create a new directory with the current date and time
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.storage_path = os.path.join('probe_data', current_time)
        os.makedirs(self.storage_path, exist_ok=True)
        self.data_file_path = os.path.join(self.storage_path, 'inference_data.pt')

    def save_data(self):
        # Save the collected data to the file
        torch.save(self.data_storage, self.data_file_path)
        # Clear the storage to avoid duplicate entries in case of multiple saves
        self.data_storage = []

    def create_state_encoders(
        self,
        obs_embed_size: int,
        prev_action_embed_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        add_prev_actions: bool,
        add_prev_action_null_token: bool,
        trainable_masked_hidden_state=False,
    ):
        get_logger().info("create_state_encoder called from parent class!")
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        get_logger().info("embeddings size fixed")
        if self.multiple_beliefs:  # multiple belief model
            get_logger().info("If multiple beliefs")
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            get_logger().info("RNNStateEncoder called for each aux_uuid")
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )
            get_logger().info("belief fusion used to create fusion model")
        
        else:  # single belief model
            get_logger().info("If single beliefs")
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )
            get_logger().info("state encoder for single belief created")

        self.state_encoders = nn.ModuleDict(state_encoders)
        get_logger().info("created state encoder from ModuleDict")
        self.belief_names = list(self.state_encoders.keys())
        get_logger().info("belief names in a list")

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            elif "goal_visual_encoder.embed_class" in key:
                new_key = key.replace(
                    "goal_visual_encoder.embed_class", "goal_visual_encoder.embed_goal"
                )
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict, **kwargs)  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = self.auxiliary_model_class(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
            for memory_key in self.belief_names
        }

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        get_logger().info(f"Not implemented error in forward_encorder")
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
        self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None
    
    def calculate_accuracy(predictions, targets, threshold=0.1):
        correct_predictions = (torch.abs(predictions - targets) < threshold).all(dim=-1)
        accuracy = correct_predictions.float().mean().item()
        return accuracy

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        # observations['target_coordinates_ind'] *= 0
        temp = observations['target_coordinates_ind2'][..., -2:].clone()
        observations['target_coordinates_ind2'][..., -2:] *= 0
        get_logger().info(f"FORWARD METHOD obs: {observations['target_coordinates_ind2']} and temp = {temp}")
        """
        target_coordinates = observations['target_coordinates_ind']
        print("Memory tensor shape:", memory.tensor(list(self.state_encoders.keys())[0]).shape)
        print("Observation 'target_coordinates_ind' shape:", observations['target_coordinates_ind'].shape)
        if 'target_coordinates_ind' in observations:
            data_pair = {
                'memory_tensor': memory.tensor(list(self.state_encoders.keys())[0]).detach().cpu(),
                'target_coordinates_ind': observations['target_coordinates_ind'].detach().cpu()
            }
            self.data_storage.append(data_pair)
        if len(self.data_storage) >= 1:
            self.save_data()
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)
        # get_logger().info(f"FORWARD METHOD obs: {obs_embeds}")

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here



        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(beliefs),
            values=self.critic(beliefs),
            extras=extras,
        )

        return actor_critic_output, memory
