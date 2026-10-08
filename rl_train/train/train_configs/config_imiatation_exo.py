from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from dataclasses import dataclass, field


@dataclass
class ExoImitationTrainSessionConfig(ImitationTrainSessionConfig):
    @dataclass
    class PolicyParams(ImitationTrainSessionConfig.PolicyParams):
        """
        ActorCriticPolicy parameters:
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
            activation_fn: type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[dict[str, Any]] = None,
        """

        # @dataclass
        # class CustomPolicyParams:
        #     reset_shared_net: bool = False
        #     reset_policy_net: bool = False
        #     reset_value_net: bool = False
        # custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

        # This actually does nothing
        @dataclass
        class CustomPolicyParams(ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams):
            human_observation_indices: list[int] = field(default_factory=list[int])
            exo_observation_indices: list[int] = field(default_factory=list[int])
            human_action_size: int = 0
            exo_action_size: int = 0

        custom_policy_params: CustomPolicyParams = field(default_factory=CustomPolicyParams)

    policy_params: PolicyParams = field(default_factory=PolicyParams)

    @dataclass
    class EnvParams(ImitationTrainSessionConfig.EnvParams):
        @dataclass
        class RewardWeights(ImitationTrainSessionConfig.EnvParams.RewardWeights):
            # Diagnostic, not a reward: the mean over both legs of whether the scripted exo controller
            # had a valid gait phase this step. Keep the weight 0. Its per-episode sum in train_log.json
            # divided by average_num_timestep is the fraction of steps the exo was actually assisting.
            exo_phase_valid: float = 0.0

        reward_keys_and_weights: RewardWeights = field(default_factory=RewardWeights)

        @dataclass
        class ExoControllerParams:
            """Parameters of the scripted exo controller named by ``env_params.device_controller``.

            The controller replaces the policy's exo torque from inside the physics loop: the policy still emits the
            exo actions, but the controller sets the exo's ctrl on every physics substep, so the policy learns muscle
            control under a fixed assistance profile. Scalars only, so that every field is overridable as
            --config.env_params.exo_controller_params.<field>.

            The spline and stride values default to the NeuMove ExoBoot's tuned hardware parameters.
            """

            right_actuator: str = "Exo_R"
            left_actuator: str = "Exo_L"

            # Torque profile knots on the full gait cycle, heel strike to heel strike (0-1), in N*m.
            rise_fraction: float = 0.278
            peak_fraction: float = 0.543
            fall_fraction: float = 0.641
            peak_torque: float = 25.0
            # The boot runs a 3 N*m bias (SPLINE_BIAS) to keep its Bowden cable taut. There is no cable in the
            # simulation, where a bias is just extra plantarflexion through stance, so it is off. With the boot's
            # knots and a 1.15 s stride, 3 N*m adds 20% to the torque impulse per stride (5.0 -> 6.0 N*m*s); set
            # 3.0 to match what the boot delivered.
            bias_torque: float = 0.0
            peak_hold_time: float = 0.0
            # Stance ends here and torque drops to zero until the next heel strike, as on the boot, whose state
            # machine hands the spline over to reel-out and swing at TOE_OFF_FRACTION. With the default knots this
            # cuts the fall short (fall_fraction is later), from 9.5 N*m (11.3 with the boot's bias). 1.0 applies
            # the spline all cycle long.
            toe_off_fraction: float = 0.60
            # No torque for this long after each heel strike: the boot reels in its cable before stance control
            # starts, commanding no spline torque meanwhile. A time, not a phase -- it did not follow step
            # frequency across speed changes. Measured on the validation log at 1.25 m/s: 150 ms left, 164 ms right
            # (this is their mean). Re-measure for other boots or sessions with tools/replay_exoboot_log.py.
            reel_in_time: float = 0.157
            # Rate of the device_controller, inside the 1200 Hz physics. 150 Hz ticks exactly every 8 substeps; the
            # boot's own loop is 175 Hz, and the spline needs no more than 150 (its timing is in seconds and phase,
            # not samples).
            controller_rate_hz: float = 150.0
            # Run the controller but apply none of its torque (device.ShadowDevice): the gait is the exo-off gait bit
            # for bit, and the controller's diagnostics still report what it would do. For judging its sensing on a
            # trained policy.
            shadow_mode: bool = False

            # Stride-average gait phase, as on the ExoBoot. Phase is invalid (no torque) until
            # num_strides_required strides in a row fall inside the duration bounds.
            num_strides_required: int = 2
            num_strides_to_average: int = 2
            min_stride_duration: float = 0.6
            max_stride_duration: float = 2.0

            # Heel strikes from foot GRF (the models have no gyro). Raw newtons from the foot + toes touch
            # sensors, which read well below bodyweight, so these are not bodyweight fractions. On the MyoAssist
            # tutorial policy walking on DephyExoBoot_L1 (tools/rollout_controllers.py), every stance peaks at
            # 717-3000 N and crosses grf_on as it starts, but a foot that scuffs the ground in swing loads it for
            # 2-30 ms at 280-700 N. No threshold separates the two; their duration does. A contact only counts
            # once it has lasted min_contact_time, dated back to its onset, and only ends once the foot has been
            # unloaded for min_unload_time (a bounce after impact does not split the stride).
            heel_strike_source: str = "vgrf"
            grf_on_newtons: float = 100.0
            grf_off_newtons: float = 25.0
            min_unload_time: float = 0.05
            min_contact_time: float = 0.05

            # For the controllers that run the boot's state machine and read its ankle encoder
            # (myoassist_utils/exo_ctrl/boot_state.py); 4PTS uses neither. Each controller's configs set the values
            # measured on the boot's logs of its own sessions; the defaults are the DL sessions'.
            # Toe-off to swing, the boot's reel-out: no torque. On the DL sessions' logs, 167-175 ms per leg.
            reel_out_time: float = 0.172
            # What the boot's ankle angle reads at the standing keyframe, per side, deg: its reading in the quiet
            # standing that starts each log (the DL sessions' mean here). Not CONFIG's <SIDE>_STANDING_ANGLE: the boot
            # reads that during its calibration with the cable taut, 1-6 deg higher.
            ankle_standing_angle_r_deg: float = -1.87
            ankle_standing_angle_l_deg: float = -10.32
            # The Dephy's resolution on the simulated boot sensors: one encoder click (360/2^14 deg) on the ankle angle,
            # and 1/8192 g and 1/32.75 deg/s on a simulated IMU.
            sensor_quantize: bool = True

        exo_controller_params: ExoControllerParams = field(default_factory=ExoControllerParams)

        # The scripted controller that drives the exo from inside the physics loop, on every physics substep: ""
        # (none: the policy's exo actions apply), "zero" (no torque), "exoboot_spline" (the ExoBoot four-point
        # spline at exo_controller_params.controller_rate_hz). Its parameters are read from exo_controller_params.
        # Needs env_id myoAssistLegImitationExoDevice-v0.
        device_controller: str = ""

    env_params: EnvParams = field(default_factory=EnvParams)


##############################################################################
