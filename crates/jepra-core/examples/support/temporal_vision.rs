#![allow(clippy::items_after_test_module)]

use jepra_core::{
    Conv2d, ConvEncoder, EmbeddingEncoder, RepresentationHealthStats, SignedMarginObjectiveConfig,
    Tensor,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const BATCH_SIZE: usize = 8;
pub const IMAGE_SIZE: usize = 8;
pub const CHANNELS: usize = 1;
pub const SQUARE_SIZE: usize = 2;
pub const SLOW_MOTION_DX: usize = 1;
pub const FAST_MOTION_DX: usize = 2;
pub const FAST_MOTION_MASS_THRESHOLD: f32 = 0.8f32 * (SQUARE_SIZE * SQUARE_SIZE) as f32;
pub const EXTRA_SQUARE_CHANCE: f64 = 0.5;
pub const MIXED_MODE_SEARCH_LIMIT: u64 = 64;
pub const MIN_MIXED_MODE_COUNT: usize = 2;
pub const VELOCITY_TRAIL_INTENSITY_RATIO: f32 = 0.35;
pub const VELOCITY_TRAIL_DECAY: f32 = 0.82;
#[allow(dead_code)]
pub const VELOCITY_BANK_CANDIDATE_DX: [isize; 2] = [1, 2];
#[allow(dead_code)]
pub const SIGNED_VELOCITY_BANK_CANDIDATE_DX: [isize; 4] = [-2, -1, 1, 2];
pub const MIN_SIGNED_MODE_COUNT: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactEncoderMode {
    Disabled,
    Base,
    Stronger,
    SignedDirection,
}

impl CompactEncoderMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "frozen(base)",
            Self::Base => "compact(base)",
            Self::Stronger => "compact(stronger)",
            Self::SignedDirection => "compact(signed-direction)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorMode {
    Baseline,
    Bottleneck,
    ResidualBottleneck,
}

impl PredictorMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Bottleneck => "bottleneck",
            Self::ResidualBottleneck => "residual-bottleneck",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalTaskMode {
    RandomSpeed,
    VelocityTrail,
    SignedVelocityTrail,
}

impl TemporalTaskMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RandomSpeed => "random-speed",
            Self::VelocityTrail => "velocity-trail",
            Self::SignedVelocityTrail => "signed-velocity-trail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalRunConfig {
    pub train_base_seed: u64,
    pub total_steps: usize,
    pub log_every: usize,
    pub encoder_learning_rate: f32,
    pub temporal_task_mode: TemporalTaskMode,
    pub compact_encoder_mode: CompactEncoderMode,
    pub predictor_mode: PredictorMode,
    pub residual_delta_scale: f32,
    pub projector_drift_weight: f32,
    pub signed_margin_weight: f32,
    pub signed_margin_config: SignedMarginObjectiveConfig,
    pub target_projection_momentum: f32,
    pub target_projection_momentum_start: f32,
    pub target_projection_momentum_end: f32,
    pub target_projection_momentum_warmup_steps: usize,
}

impl TemporalRunConfig {
    pub fn from_args(
        default_train_base_seed: u64,
        default_total_steps: usize,
        default_log_every: usize,
        default_encoder_learning_rate: f32,
    ) -> Self {
        let args: Vec<String> = std::env::args().collect();

        let train_base_seed = parse_u64_arg(&args, "--train-base-seed")
            .or_else(|| parse_u64_arg(&args, "--seed"))
            .unwrap_or(default_train_base_seed);

        let total_steps = parse_usize_arg(&args, "--train-steps")
            .or_else(|| parse_usize_arg(&args, "--steps"))
            .or_else(|| Some(training_steps(default_total_steps)))
            .unwrap_or(default_total_steps);

        let log_every = parse_usize_arg(&args, "--log-every")
            .or_else(|| parse_usize_arg(&args, "--log"))
            .unwrap_or(default_log_every);

        let encoder_learning_rate = parse_f32_arg(&args, "--encoder-lr")
            .or_else(|| parse_f32_arg(&args, "--encoder-learning-rate"))
            .or_else(|| parse_f32_arg_env("JEPRA_ENCODER_LR"))
            .unwrap_or(default_encoder_learning_rate);
        let target_projection_momentum_end = parse_f32_arg(&args, "--target-momentum-end")
            .or_else(|| parse_f32_arg(&args, "--target-projection-momentum-end"))
            .or_else(|| parse_f32_arg(&args, "--target-momentum"))
            .or_else(|| parse_f32_arg(&args, "--target-projection-momentum"))
            .or_else(|| parse_f32_arg_env("JEPRA_TARGET_MOMENTUM"))
            .unwrap_or(1.0);
        let target_projection_momentum_start = parse_f32_arg(&args, "--target-momentum-start")
            .unwrap_or(target_projection_momentum_end);
        let target_projection_momentum_warmup_steps =
            parse_usize_arg(&args, "--target-momentum-warmup-steps")
                .or_else(|| parse_usize_arg(&args, "--target-projection-warmup-steps"))
                .unwrap_or(0);
        let compact_encoder_mode = compact_encoder_mode_from_args(&args);
        let predictor_mode = predictor_mode_from_args(&args);
        let temporal_task_mode = temporal_task_mode_from_args(&args);
        let residual_delta_scale = residual_delta_scale_from_args(&args);
        let projector_drift_weight = projector_drift_weight_from_args(&args);
        let signed_margin_weight = signed_margin_weight_from_args(&args);
        let signed_margin_config = signed_margin_config_from_args(&args);
        assert!(
            signed_margin_weight == 0.0
                || temporal_task_mode == TemporalTaskMode::SignedVelocityTrail,
            "signed margin objective is only supported for signed-velocity-trail task"
        );
        assert_target_projection_momentum(target_projection_momentum_end);
        assert_target_projection_momentum(target_projection_momentum_start);

        assert!(
            total_steps > 0,
            "train steps must be greater than 0, got {}",
            total_steps
        );
        assert!(
            log_every > 0,
            "log_every must be greater than 0, got {}",
            log_every
        );

        Self {
            train_base_seed,
            total_steps,
            log_every,
            encoder_learning_rate,
            temporal_task_mode,
            compact_encoder_mode,
            predictor_mode,
            residual_delta_scale,
            projector_drift_weight,
            signed_margin_weight,
            signed_margin_config,
            target_projection_momentum: target_projection_momentum_end,
            target_projection_momentum_start,
            target_projection_momentum_end,
            target_projection_momentum_warmup_steps,
        }
    }

    #[allow(dead_code)]
    pub fn target_projection_momentum_at_step(&self, step: usize) -> f32 {
        if self.target_projection_momentum_warmup_steps == 0 {
            return self.target_projection_momentum_end;
        }

        let clamped_step = step.min(self.target_projection_momentum_warmup_steps) as f32;
        let alpha = clamped_step / self.target_projection_momentum_warmup_steps as f32;
        self.target_projection_momentum_start
            + alpha * (self.target_projection_momentum_end - self.target_projection_momentum_start)
    }
}

fn assert_target_projection_momentum(target_projection_momentum: f32) {
    assert!(
        (0.0..=1.0).contains(&target_projection_momentum),
        "target projection momentum must be in [0.0, 1.0], got {}",
        target_projection_momentum
    );
}

fn compact_encoder_mode_from_args(args: &[String]) -> CompactEncoderMode {
    let compact_mode_from_flag = parse_compact_encoder_mode_flag(args);
    let compact_encoder_enabled = args.iter().any(|arg| arg == "--compact-encoder");

    match (compact_encoder_enabled, compact_mode_from_flag) {
        (true, Some(mode)) => mode,
        (true, None) => CompactEncoderMode::Stronger,
        (false, Some(mode)) => mode,
        (false, None) => CompactEncoderMode::Disabled,
    }
}

fn parse_compact_encoder_mode_flag(args: &[String]) -> Option<CompactEncoderMode> {
    parse_arg_value(args, "--compact-encoder-mode").map(|raw_mode| match raw_mode {
        "base" => CompactEncoderMode::Base,
        "stronger" => CompactEncoderMode::Stronger,
        "signed-direction" => CompactEncoderMode::SignedDirection,
        _ => panic!(
            "unsupported value for --compact-encoder-mode: {} (expected base|stronger|signed-direction)",
            raw_mode
        ),
    })
}

fn predictor_mode_from_args(args: &[String]) -> PredictorMode {
    parse_arg_value(args, "--predictor-mode")
        .map(|raw_mode| match raw_mode {
            "baseline" => PredictorMode::Baseline,
            "bottleneck" => PredictorMode::Bottleneck,
            "residual-bottleneck" => PredictorMode::ResidualBottleneck,
            _ => panic!(
                "unsupported value for --predictor-mode: {} (expected baseline|bottleneck|residual-bottleneck)",
                raw_mode
            ),
        })
        .unwrap_or(PredictorMode::Baseline)
}

fn temporal_task_mode_from_args(args: &[String]) -> TemporalTaskMode {
    parse_arg_value(args, "--temporal-task")
        .or_else(|| parse_arg_value(args, "--task"))
        .map(|raw_mode| parse_temporal_task_mode(raw_mode, "--temporal-task"))
        .or_else(|| {
            std::env::var("JEPRA_TEMPORAL_TASK")
                .or_else(|_| std::env::var("JEPRA_TASK"))
                .ok()
                .map(|raw_mode| parse_temporal_task_mode(&raw_mode, "JEPRA_TEMPORAL_TASK"))
        })
        .unwrap_or(TemporalTaskMode::RandomSpeed)
}

fn parse_temporal_task_mode(raw_mode: &str, source: &str) -> TemporalTaskMode {
    match raw_mode {
        "random-speed" | "current-squares" => TemporalTaskMode::RandomSpeed,
        "velocity-trail" | "harder-squares" => TemporalTaskMode::VelocityTrail,
        "signed-velocity-trail" | "signed-trail" => TemporalTaskMode::SignedVelocityTrail,
        _ => panic!(
            "unsupported value for {}: {} (expected random-speed|velocity-trail|signed-velocity-trail)",
            source, raw_mode
        ),
    }
}

fn residual_delta_scale_from_args(args: &[String]) -> f32 {
    parse_arg_value(args, "--residual-delta-scale")
        .map(|value| parse_finite_nonnegative_f32(value, "--residual-delta-scale"))
        .or_else(|| {
            std::env::var("JEPRA_RESIDUAL_DELTA_SCALE")
                .ok()
                .map(|value| parse_finite_nonnegative_f32(&value, "JEPRA_RESIDUAL_DELTA_SCALE"))
        })
        .unwrap_or(1.0)
}

fn projector_drift_weight_from_args(args: &[String]) -> f32 {
    parse_arg_value(args, "--projector-drift-weight")
        .or_else(|| parse_arg_value(args, "--projector-anchor-weight"))
        .map(|value| parse_finite_nonnegative_f32(value, "--projector-drift-weight"))
        .or_else(|| {
            std::env::var("JEPRA_PROJECTOR_DRIFT_WEIGHT")
                .or_else(|_| std::env::var("JEPRA_PROJECTOR_ANCHOR_WEIGHT"))
                .ok()
                .map(|value| parse_finite_nonnegative_f32(&value, "JEPRA_PROJECTOR_DRIFT_WEIGHT"))
        })
        .unwrap_or(0.0)
}

fn signed_margin_weight_from_args(args: &[String]) -> f32 {
    parse_arg_value(args, "--signed-margin-weight")
        .or_else(|| parse_arg_value(args, "--margin-objective-weight"))
        .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-weight"))
        .or_else(|| {
            std::env::var("JEPRA_SIGNED_MARGIN_WEIGHT")
                .or_else(|_| std::env::var("JEPRA_MARGIN_OBJECTIVE_WEIGHT"))
                .ok()
                .map(|value| parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_WEIGHT"))
        })
        .unwrap_or(0.0)
}

fn signed_margin_config_from_args(args: &[String]) -> SignedMarginObjectiveConfig {
    let config = SignedMarginObjectiveConfig {
        bank_gap: parse_arg_value(args, "--signed-margin-bank-gap")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-bank-gap"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_BANK_GAP")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_BANK_GAP")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().bank_gap),
        sign_gap: parse_arg_value(args, "--signed-margin-sign-gap")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-sign-gap"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_SIGN_GAP")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_SIGN_GAP")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().sign_gap),
        speed_gap: parse_arg_value(args, "--signed-margin-speed-gap")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-speed-gap"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_SPEED_GAP")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_SPEED_GAP")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().speed_gap),
        bank_weight: parse_arg_value(args, "--signed-margin-bank-weight")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-bank-weight"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_BANK_WEIGHT")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_BANK_WEIGHT")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().bank_weight),
        sign_weight: parse_arg_value(args, "--signed-margin-sign-weight")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-sign-weight"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_SIGN_WEIGHT")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_SIGN_WEIGHT")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().sign_weight),
        speed_weight: parse_arg_value(args, "--signed-margin-speed-weight")
            .map(|value| parse_finite_nonnegative_f32(value, "--signed-margin-speed-weight"))
            .or_else(|| {
                std::env::var("JEPRA_SIGNED_MARGIN_SPEED_WEIGHT")
                    .ok()
                    .map(|value| {
                        parse_finite_nonnegative_f32(&value, "JEPRA_SIGNED_MARGIN_SPEED_WEIGHT")
                    })
            })
            .unwrap_or(SignedMarginObjectiveConfig::default().speed_weight),
    };
    config.assert_valid();
    config
}

pub fn training_steps(default_steps: usize) -> usize {
    std::env::var("JEPRA_TRAIN_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&steps| steps > 0)
        .unwrap_or(default_steps)
}

fn parse_arg_value<'a>(args: &'a [String], flag: &'a str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        if window[0] == flag {
            Some(window[1].as_str())
        } else {
            None
        }
    })
}

fn parse_u64_arg(args: &[String], flag: &str) -> Option<u64> {
    parse_arg_value(args, flag).and_then(|value| value.parse::<u64>().ok())
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|&value| value.is_finite() && value >= 0.0)
}

fn parse_f32_arg_env(flag: &str) -> Option<f32> {
    std::env::var(flag)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_finite_nonnegative_f32(value: &str, source: &str) -> f32 {
    let parsed = value
        .parse::<f32>()
        .unwrap_or_else(|_| panic!("{source} must be a finite non-negative float, got {value}"));
    assert!(
        parsed.is_finite() && parsed >= 0.0,
        "{} must be finite and non-negative, got {}",
        source,
        parsed
    );
    parsed
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalExperimentSummary {
    pub config: TemporalRunConfig,
    pub initial_train_loss: f32,
    pub final_train_loss: f32,
    pub initial_validation_loss: f32,
    pub final_validation_loss: f32,
}

impl TemporalExperimentSummary {
    pub fn train_delta(&self) -> f32 {
        self.final_train_loss - self.initial_train_loss
    }

    pub fn validation_delta(&self) -> f32 {
        self.final_validation_loss - self.initial_validation_loss
    }

    pub fn train_improved(&self) -> bool {
        self.final_train_loss < self.initial_train_loss
    }

    pub fn validation_improved(&self) -> bool {
        self.final_validation_loss < self.initial_validation_loss
    }
}

pub fn run_temporal_experiment_with_summary<TModel, TStep, TValue>(
    config: TemporalRunConfig,
    model: &mut TModel,
    initial_train_loss: f32,
    initial_validation_loss: f32,
    mut train_step: TStep,
    mut validation_loss: TValue,
) -> TemporalExperimentSummary
where
    TStep: FnMut(&mut TModel, usize, bool) -> f32,
    TValue: FnMut(&TModel) -> f32,
{
    let mut final_train_loss = initial_train_loss;

    run_temporal_training_loop(config.total_steps, config.log_every, |step, should_log| {
        final_train_loss = train_step(model, step, should_log);
    });

    let final_validation_loss = validation_loss(model);

    TemporalExperimentSummary {
        config,
        initial_train_loss,
        final_train_loss,
        initial_validation_loss,
        final_validation_loss,
    }
}

#[allow(dead_code)]
pub fn print_temporal_experiment_summary(tag: &str, summary: &TemporalExperimentSummary) {
    println!(
        "{} run summary | steps {} | train {:.6} -> {:.6} (Δ {:+.6}, improved={}) | val {:.6} -> {:.6} (Δ {:+.6}, improved={})",
        tag,
        summary.config.total_steps,
        summary.initial_train_loss,
        summary.final_train_loss,
        summary.train_delta(),
        summary.train_improved(),
        summary.initial_validation_loss,
        summary.final_validation_loss,
        summary.validation_delta(),
        summary.validation_improved()
    );
}

pub fn assert_temporal_experiment_improved(
    experiment_tag: &str,
    initial_train_loss: f32,
    final_train_loss: f32,
    initial_validation_loss: f32,
    final_validation_loss: f32,
    train_max_reduction_ratio: f32,
    validation_max_reduction_ratio: f32,
) {
    assert!(
        final_train_loss < initial_train_loss,
        "{} probe train loss did not improve: {:.6} -> {:.6}",
        experiment_tag,
        initial_train_loss,
        final_train_loss
    );
    assert!(
        final_validation_loss < initial_validation_loss,
        "{} validation loss did not improve: {:.6} -> {:.6}",
        experiment_tag,
        initial_validation_loss,
        final_validation_loss
    );
    assert!(
        final_train_loss < initial_train_loss * train_max_reduction_ratio,
        "{} train loss stayed too high: {:.6} >= {:.6}",
        experiment_tag,
        final_train_loss,
        initial_train_loss * train_max_reduction_ratio
    );
    assert!(
        final_validation_loss < initial_validation_loss * validation_max_reduction_ratio,
        "{} validation loss stayed too high: {:.6} >= {:.6}",
        experiment_tag,
        final_validation_loss,
        initial_validation_loss * validation_max_reduction_ratio
    );
}

#[cfg(test)]
mod temporal_vision_config_tests {
    use super::{
        CompactEncoderMode, PredictorMode, SignedMarginObjectiveConfig, TemporalRunConfig,
        TemporalTaskMode, compact_encoder_mode_from_args, predictor_mode_from_args,
        projector_drift_weight_from_args, residual_delta_scale_from_args,
        signed_margin_config_from_args, signed_margin_weight_from_args,
        temporal_task_mode_from_args,
    };

    fn args_with(values: &[&str]) -> Vec<String> {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    }

    #[test]
    fn compact_encoder_mode_defaults_to_disabled() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&[])),
            CompactEncoderMode::Disabled
        );
    }

    #[test]
    fn compact_encoder_mode_switches_on_compact_flag() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&["--compact-encoder"])),
            CompactEncoderMode::Stronger
        );
    }

    #[test]
    fn compact_encoder_mode_supports_stronger_mode_flag() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&["--compact-encoder-mode", "stronger"])),
            CompactEncoderMode::Stronger
        );
    }

    #[test]
    fn compact_encoder_flag_uses_stronger_mode() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&[
                "--compact-encoder",
                "--compact-encoder-mode",
                "stronger"
            ])),
            CompactEncoderMode::Stronger
        );
    }

    #[test]
    fn compact_encoder_base_mode_is_explicit() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&[
                "--compact-encoder",
                "--compact-encoder-mode",
                "base"
            ])),
            CompactEncoderMode::Base
        );
    }

    #[test]
    fn compact_encoder_signed_direction_mode_is_explicit() {
        assert_eq!(
            compact_encoder_mode_from_args(&args_with(&[
                "--compact-encoder-mode",
                "signed-direction"
            ])),
            CompactEncoderMode::SignedDirection
        );
    }

    #[test]
    #[should_panic(expected = "unsupported value for --compact-encoder-mode")]
    fn compact_encoder_mode_panics_on_invalid_value() {
        compact_encoder_mode_from_args(&args_with(&["--compact-encoder-mode", "mega"]));
    }

    #[test]
    fn predictor_mode_defaults_to_baseline() {
        assert_eq!(
            predictor_mode_from_args(&args_with(&[])),
            PredictorMode::Baseline
        );
    }

    #[test]
    fn predictor_mode_parses_bottleneck() {
        assert_eq!(
            predictor_mode_from_args(&args_with(&["--predictor-mode", "bottleneck"])),
            PredictorMode::Bottleneck
        );
    }

    #[test]
    fn predictor_mode_parses_residual_bottleneck() {
        assert_eq!(
            predictor_mode_from_args(&args_with(&["--predictor-mode", "residual-bottleneck"])),
            PredictorMode::ResidualBottleneck
        );
    }

    #[test]
    #[should_panic(expected = "unsupported value for --predictor-mode")]
    fn predictor_mode_panics_on_invalid_value() {
        predictor_mode_from_args(&args_with(&["--predictor-mode", "wide"]));
    }

    #[test]
    fn temporal_task_mode_defaults_to_random_speed() {
        assert_eq!(
            temporal_task_mode_from_args(&args_with(&[])),
            TemporalTaskMode::RandomSpeed
        );
    }

    #[test]
    fn temporal_task_mode_parses_velocity_trail() {
        assert_eq!(
            temporal_task_mode_from_args(&args_with(&["--temporal-task", "velocity-trail"])),
            TemporalTaskMode::VelocityTrail
        );
    }

    #[test]
    fn temporal_task_mode_parses_signed_velocity_trail() {
        assert_eq!(
            temporal_task_mode_from_args(&args_with(&["--temporal-task", "signed-velocity-trail"])),
            TemporalTaskMode::SignedVelocityTrail
        );
    }

    #[test]
    fn temporal_task_mode_keeps_compatibility_aliases() {
        assert_eq!(
            temporal_task_mode_from_args(&args_with(&["--task", "harder-squares"])),
            TemporalTaskMode::VelocityTrail
        );
        assert_eq!(
            temporal_task_mode_from_args(&args_with(&["--task", "current-squares"])),
            TemporalTaskMode::RandomSpeed
        );
    }

    #[test]
    #[should_panic(expected = "unsupported value for --temporal-task")]
    fn temporal_task_mode_panics_on_invalid_value() {
        temporal_task_mode_from_args(&args_with(&["--temporal-task", "bounce"]));
    }

    #[test]
    fn residual_delta_scale_defaults_to_unscaled_delta() {
        assert!((residual_delta_scale_from_args(&args_with(&[])) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn residual_delta_scale_parses_explicit_scale() {
        assert!(
            (residual_delta_scale_from_args(&args_with(&["--residual-delta-scale", "0.25"]))
                - 0.25)
                .abs()
                < 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "--residual-delta-scale must be finite and non-negative")]
    fn residual_delta_scale_panics_on_negative_value() {
        residual_delta_scale_from_args(&args_with(&["--residual-delta-scale", "-0.1"]));
    }

    #[test]
    fn projector_drift_weight_defaults_to_disabled() {
        assert!((projector_drift_weight_from_args(&args_with(&[])) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn projector_drift_weight_parses_explicit_weight() {
        assert!(
            (projector_drift_weight_from_args(&args_with(&["--projector-drift-weight", "2.5"]))
                - 2.5)
                .abs()
                < 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "--projector-drift-weight must be finite and non-negative")]
    fn projector_drift_weight_panics_on_negative_value() {
        projector_drift_weight_from_args(&args_with(&["--projector-drift-weight", "-0.1"]));
    }

    #[test]
    fn signed_margin_weight_defaults_to_disabled() {
        assert!((signed_margin_weight_from_args(&args_with(&[])) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn signed_margin_weight_parses_explicit_weight() {
        assert!(
            (signed_margin_weight_from_args(&args_with(&["--signed-margin-weight", "0.03"]))
                - 0.03)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn signed_margin_config_parses_explicit_gaps_and_weights() {
        let config = signed_margin_config_from_args(&args_with(&[
            "--signed-margin-bank-gap",
            "0.1",
            "--signed-margin-sign-gap",
            "0.2",
            "--signed-margin-speed-gap",
            "0.3",
            "--signed-margin-bank-weight",
            "0.4",
            "--signed-margin-sign-weight",
            "0.5",
            "--signed-margin-speed-weight",
            "0.6",
        ]));

        assert!((config.bank_gap - 0.1).abs() < 1e-6);
        assert!((config.sign_gap - 0.2).abs() < 1e-6);
        assert!((config.speed_gap - 0.3).abs() < 1e-6);
        assert!((config.bank_weight - 0.4).abs() < 1e-6);
        assert!((config.sign_weight - 0.5).abs() < 1e-6);
        assert!((config.speed_weight - 0.6).abs() < 1e-6);
    }

    #[test]
    fn target_projection_momentum_warms_linearly_to_end() {
        let config = TemporalRunConfig {
            train_base_seed: 0,
            total_steps: 10,
            log_every: 1,
            encoder_learning_rate: 0.0,
            temporal_task_mode: TemporalTaskMode::RandomSpeed,
            compact_encoder_mode: CompactEncoderMode::Disabled,
            predictor_mode: PredictorMode::Baseline,
            residual_delta_scale: 1.0,
            projector_drift_weight: 0.0,
            signed_margin_weight: 0.0,
            signed_margin_config: SignedMarginObjectiveConfig::default(),
            target_projection_momentum: 0.5,
            target_projection_momentum_start: 1.0,
            target_projection_momentum_end: 0.5,
            target_projection_momentum_warmup_steps: 2,
        };

        assert!((config.target_projection_momentum_at_step(0) - 1.0).abs() < 1e-6);
        assert!((config.target_projection_momentum_at_step(1) - 0.75).abs() < 1e-6);
        assert!((config.target_projection_momentum_at_step(2) - 0.5).abs() < 1e-6);
        assert!((config.target_projection_momentum_at_step(10) - 0.5).abs() < 1e-6);
    }
}

pub fn motion_dx_for_pair(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> usize {
    match signed_motion_dx_for_pair(x_t, x_t1, sample) {
        1 => SLOW_MOTION_DX,
        2 => FAST_MOTION_DX,
        dx => panic!("unexpected positive motion delta {}", dx),
    }
}

pub fn motion_dx_for_sample(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> usize {
    motion_dx_for_pair(x_t, x_t1, sample)
}

pub fn signed_motion_dx_for_pair(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> isize {
    let center_x_t = square_center_x(x_t, sample);
    let center_x_t1 = square_center_x(x_t1, sample);
    let delta_x = (center_x_t1 - center_x_t).round();

    for candidate_dx in SIGNED_VELOCITY_BANK_CANDIDATE_DX {
        if (delta_x - candidate_dx as f32).abs() < 1e-6 {
            return candidate_dx;
        }
    }

    panic!("unexpected motion delta {:.6}", delta_x);
}

pub fn signed_motion_dx_for_sample(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> isize {
    signed_motion_dx_for_pair(x_t, x_t1, sample)
}

fn random_motion_dx(rng: &mut StdRng) -> usize {
    if rng.gen_bool(0.5) {
        SLOW_MOTION_DX
    } else {
        FAST_MOTION_DX
    }
}

fn balanced_signed_motion_dx_sequence(batch_size: usize, rng: &mut StdRng) -> Vec<isize> {
    let mut motion_dx = Vec::with_capacity(batch_size);

    while motion_dx.len() < batch_size {
        for candidate_dx in SIGNED_VELOCITY_BANK_CANDIDATE_DX {
            if motion_dx.len() == batch_size {
                break;
            }
            motion_dx.push(candidate_dx);
        }
    }

    for index in (1..motion_dx.len()).rev() {
        let swap_index = rng.gen_range(0..=index);
        motion_dx.swap(index, swap_index);
    }

    motion_dx
}

#[allow(dead_code)]
pub fn make_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    make_temporal_batch_for_task(batch_size, seed, TemporalTaskMode::RandomSpeed)
}

pub fn make_temporal_batch_for_task(
    batch_size: usize,
    seed: u64,
    temporal_task_mode: TemporalTaskMode,
) -> (Tensor, Tensor) {
    match temporal_task_mode {
        TemporalTaskMode::RandomSpeed => make_random_speed_temporal_batch(batch_size, seed),
        TemporalTaskMode::VelocityTrail => make_velocity_trail_temporal_batch(batch_size, seed),
        TemporalTaskMode::SignedVelocityTrail => {
            make_signed_velocity_trail_temporal_batch(batch_size, seed)
        }
    }
}

fn make_random_speed_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_t = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    let mut x_t1 = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

    let max_row = IMAGE_SIZE - SQUARE_SIZE;
    let max_col_t = IMAGE_SIZE - SQUARE_SIZE - FAST_MOTION_DX;

    for sample in 0..batch_size {
        let row = rng.gen_range(0..=max_row);
        let col_t = rng.gen_range(0..=max_col_t);
        let intensity_t = rng.gen_range(0.65f32..0.95f32);
        let motion_dx = random_motion_dx(&mut rng);
        let col_t1 = col_t + motion_dx;
        let intensity_t1 = (0.9f32 * intensity_t + 0.05f32).clamp(0.5f32, 1.0f32);

        draw_square(&mut x_t, sample, row, col_t, intensity_t);
        draw_square(&mut x_t1, sample, row, col_t1, intensity_t1);

        if rng.gen_bool(EXTRA_SQUARE_CHANCE) {
            let mut secondary_row = rng.gen_range(0..=max_row);
            let mut secondary_col_t = rng.gen_range(0..=max_col_t);
            let mut attempts = 0usize;

            while squares_overlap(row, col_t, secondary_row, secondary_col_t) && attempts < 16 {
                secondary_row = rng.gen_range(0..=max_row);
                secondary_col_t = rng.gen_range(0..=max_col_t);
                attempts += 1;
            }

            let secondary_intensity_t = intensity_t;
            let secondary_intensity_t1 =
                (0.9f32 * secondary_intensity_t + 0.05f32).clamp(0.5f32, 1.0f32);

            draw_square(
                &mut x_t,
                sample,
                secondary_row,
                secondary_col_t,
                secondary_intensity_t,
            );
            draw_square(
                &mut x_t1,
                sample,
                secondary_row,
                secondary_col_t + motion_dx,
                secondary_intensity_t1,
            );
        }
    }

    (x_t, x_t1)
}

fn make_velocity_trail_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_t = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    let mut x_t1 = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

    let max_row = IMAGE_SIZE - SQUARE_SIZE;
    let min_col_t = FAST_MOTION_DX;
    let max_col_t = IMAGE_SIZE - SQUARE_SIZE - FAST_MOTION_DX;

    for sample in 0..batch_size {
        let row = rng.gen_range(0..=max_row);
        let col_t = rng.gen_range(min_col_t..=max_col_t);
        let intensity_t = rng.gen_range(0.65f32..0.95f32);
        let motion_dx = random_motion_dx(&mut rng);
        let trail_col_t = col_t - motion_dx;
        let col_t1 = col_t + motion_dx;
        let trail_intensity_t = intensity_t * VELOCITY_TRAIL_INTENSITY_RATIO;
        let intensity_t1 = intensity_t * VELOCITY_TRAIL_DECAY;
        let trail_intensity_t1 = trail_intensity_t * VELOCITY_TRAIL_DECAY;

        add_square(&mut x_t, sample, row, trail_col_t, trail_intensity_t);
        add_square(&mut x_t, sample, row, col_t, intensity_t);
        add_square(&mut x_t1, sample, row, col_t, trail_intensity_t1);
        add_square(&mut x_t1, sample, row, col_t1, intensity_t1);
    }

    (x_t, x_t1)
}

fn make_signed_velocity_trail_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let signed_motion_dx = balanced_signed_motion_dx_sequence(batch_size, &mut rng);
    let mut x_t = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    let mut x_t1 = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

    let max_row = IMAGE_SIZE - SQUARE_SIZE;
    let min_col_t = FAST_MOTION_DX;
    let max_col_t = IMAGE_SIZE - SQUARE_SIZE - FAST_MOTION_DX;

    for sample in 0..batch_size {
        let row = rng.gen_range(0..=max_row);
        let col_t = rng.gen_range(min_col_t..=max_col_t);
        let intensity_t = rng.gen_range(0.65f32..0.95f32);
        let motion_dx = signed_motion_dx[sample];
        let trail_col_t = (col_t as isize - motion_dx) as usize;
        let col_t1 = (col_t as isize + motion_dx) as usize;
        let trail_intensity_t = intensity_t * VELOCITY_TRAIL_INTENSITY_RATIO;
        let intensity_t1 = intensity_t * VELOCITY_TRAIL_DECAY;
        let trail_intensity_t1 = trail_intensity_t * VELOCITY_TRAIL_DECAY;

        add_square(&mut x_t, sample, row, trail_col_t, trail_intensity_t);
        add_square(&mut x_t, sample, row, col_t, intensity_t);
        add_square(&mut x_t1, sample, row, col_t, trail_intensity_t1);
        add_square(&mut x_t1, sample, row, col_t1, intensity_t1);
    }

    (x_t, x_t1)
}

#[allow(dead_code)]
pub fn make_velocity_trail_candidate_target_batch(x_t: &Tensor, candidate_dx: isize) -> Tensor {
    make_shifted_velocity_trail_candidate_target_batch(
        x_t,
        candidate_dx,
        &VELOCITY_BANK_CANDIDATE_DX,
        "velocity-trail",
    )
}

#[allow(dead_code)]
pub fn make_signed_velocity_trail_candidate_target_batch(
    x_t: &Tensor,
    candidate_dx: isize,
) -> Tensor {
    make_shifted_velocity_trail_candidate_target_batch(
        x_t,
        candidate_dx,
        &SIGNED_VELOCITY_BANK_CANDIDATE_DX,
        "signed-velocity-trail",
    )
}

fn make_shifted_velocity_trail_candidate_target_batch(
    x_t: &Tensor,
    candidate_dx: isize,
    allowed_candidate_dx: &[isize],
    task_name: &str,
) -> Tensor {
    assert!(
        allowed_candidate_dx.contains(&candidate_dx),
        "{} candidate dx must be one of {:?}, got {}",
        task_name,
        allowed_candidate_dx,
        candidate_dx
    );
    assert!(
        x_t.shape.len() == 4,
        "{} candidate target expects rank-4 input, got {:?}",
        task_name,
        x_t.shape
    );
    assert!(
        x_t.shape[1] == CHANNELS && x_t.shape[2] == IMAGE_SIZE && x_t.shape[3] == IMAGE_SIZE,
        "{} candidate target expects shape [batch, {}, {}, {}], got {:?}",
        task_name,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        x_t.shape
    );

    let batch_size = x_t.shape[0];
    let mut candidate = Tensor::zeros(x_t.shape.clone());

    for sample in 0..batch_size {
        for row in 0..IMAGE_SIZE {
            for col in 0..IMAGE_SIZE {
                let shifted_col = col as isize + candidate_dx;
                if !(0..IMAGE_SIZE as isize).contains(&shifted_col) {
                    continue;
                }

                let value = x_t.get(&[sample, 0, row, col]);
                candidate.set(
                    &[sample, 0, row, shifted_col as usize],
                    value * VELOCITY_TRAIL_DECAY,
                );
            }
        }
    }

    candidate
}

#[allow(dead_code)]
pub fn make_train_batch(train_base_seed: u64, step: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, train_base_seed + step)
}

pub fn make_train_batch_for_task(
    train_base_seed: u64,
    step: u64,
    temporal_task_mode: TemporalTaskMode,
) -> (Tensor, Tensor) {
    make_temporal_batch_for_task(BATCH_SIZE, train_base_seed + step, temporal_task_mode)
}

pub fn make_train_batch_for_config(config: TemporalRunConfig, step: u64) -> (Tensor, Tensor) {
    make_train_batch_for_task(config.train_base_seed, step, config.temporal_task_mode)
}

#[allow(dead_code)]
pub fn make_validation_batch(validation_base_seed: u64, batch_idx: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, validation_base_seed + batch_idx)
}

pub fn make_validation_batch_for_task(
    validation_base_seed: u64,
    batch_idx: u64,
    temporal_task_mode: TemporalTaskMode,
) -> (Tensor, Tensor) {
    make_temporal_batch_for_task(
        BATCH_SIZE,
        validation_base_seed + batch_idx,
        temporal_task_mode,
    )
}

pub fn make_validation_batch_for_config(
    config: TemporalRunConfig,
    validation_base_seed: u64,
    batch_idx: u64,
) -> (Tensor, Tensor) {
    make_validation_batch_for_task(validation_base_seed, batch_idx, config.temporal_task_mode)
}

pub fn motion_mode_counts(x_t: &Tensor, x_t1: &Tensor) -> (usize, usize) {
    let mut slow_count = 0;
    let mut fast_count = 0;

    for sample in 0..BATCH_SIZE {
        match motion_dx_for_sample(x_t, x_t1, sample) {
            SLOW_MOTION_DX => slow_count += 1,
            FAST_MOTION_DX => fast_count += 1,
            dx => panic!("unexpected motion dx {}", dx),
        }
    }

    (slow_count, fast_count)
}

pub fn signed_motion_mode_counts(x_t: &Tensor, x_t1: &Tensor) -> (usize, usize, usize, usize) {
    let mut negative_fast_count = 0;
    let mut negative_slow_count = 0;
    let mut positive_slow_count = 0;
    let mut positive_fast_count = 0;

    for sample in 0..BATCH_SIZE {
        match signed_motion_dx_for_sample(x_t, x_t1, sample) {
            -2 => negative_fast_count += 1,
            -1 => negative_slow_count += 1,
            1 => positive_slow_count += 1,
            2 => positive_fast_count += 1,
            dx => panic!("unexpected signed motion dx {}", dx),
        }
    }

    (
        negative_fast_count,
        negative_slow_count,
        positive_slow_count,
        positive_fast_count,
    )
}

#[cfg(test)]
#[allow(dead_code)]
pub fn batch_has_motion_mode(x_t: &Tensor, x_t1: &Tensor, motion_dx: usize) -> bool {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);

    match motion_dx {
        SLOW_MOTION_DX => slow_count > 0,
        FAST_MOTION_DX => fast_count > 0,
        _ => false,
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub fn batch_has_both_motion_modes(x_t: &Tensor, x_t1: &Tensor) -> bool {
    batch_has_motion_mode(x_t, x_t1, SLOW_MOTION_DX)
        && batch_has_motion_mode(x_t, x_t1, FAST_MOTION_DX)
}

pub fn batch_has_min_motion_mode_counts(
    x_t: &Tensor,
    x_t1: &Tensor,
    min_slow_count: usize,
    min_fast_count: usize,
) -> bool {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);
    slow_count >= min_slow_count && fast_count >= min_fast_count
}

pub fn batch_has_min_signed_motion_mode_counts(
    x_t: &Tensor,
    x_t1: &Tensor,
    min_mode_count: usize,
) -> bool {
    let (negative_fast_count, negative_slow_count, positive_slow_count, positive_fast_count) =
        signed_motion_mode_counts(x_t, x_t1);
    negative_fast_count >= min_mode_count
        && negative_slow_count >= min_mode_count
        && positive_slow_count >= min_mode_count
        && positive_fast_count >= min_mode_count
}

pub fn batch_has_required_motion_mode_counts_for_task(
    x_t: &Tensor,
    x_t1: &Tensor,
    temporal_task_mode: TemporalTaskMode,
) -> bool {
    match temporal_task_mode {
        TemporalTaskMode::SignedVelocityTrail => {
            batch_has_min_signed_motion_mode_counts(x_t, x_t1, MIN_SIGNED_MODE_COUNT)
        }
        TemporalTaskMode::RandomSpeed | TemporalTaskMode::VelocityTrail => {
            batch_has_min_motion_mode_counts(x_t, x_t1, MIN_MIXED_MODE_COUNT, MIN_MIXED_MODE_COUNT)
        }
    }
}

#[allow(dead_code)]
pub fn make_validation_batch_with_both_motion_modes(
    validation_base_seed: u64,
    start_batch_idx: u64,
) -> (Tensor, Tensor, u64) {
    make_validation_batch_with_both_motion_modes_for_task(
        validation_base_seed,
        start_batch_idx,
        TemporalTaskMode::RandomSpeed,
    )
}

pub fn make_validation_batch_with_both_motion_modes_for_task(
    validation_base_seed: u64,
    start_batch_idx: u64,
    temporal_task_mode: TemporalTaskMode,
) -> (Tensor, Tensor, u64) {
    if temporal_task_mode == TemporalTaskMode::SignedVelocityTrail {
        return make_validation_batch_with_required_motion_modes_for_task(
            validation_base_seed,
            start_batch_idx,
            temporal_task_mode,
        );
    }

    for offset in 0..MIXED_MODE_SEARCH_LIMIT {
        let batch_idx = start_batch_idx + offset;
        let seed = validation_base_seed + batch_idx;
        let (x_t, x_t1) = make_temporal_batch_for_task(BATCH_SIZE, seed, temporal_task_mode);

        if batch_has_min_motion_mode_counts(&x_t, &x_t1, MIN_MIXED_MODE_COUNT, MIN_MIXED_MODE_COUNT)
        {
            return (x_t, x_t1, seed);
        }
    }

    panic!(
        "did not find a mixed-mode validation batch with at least {} slow and {} fast samples within {} seeds from base {} and start batch {}",
        MIN_MIXED_MODE_COUNT,
        MIN_MIXED_MODE_COUNT,
        MIXED_MODE_SEARCH_LIMIT,
        validation_base_seed,
        start_batch_idx
    );
}

pub fn make_validation_batch_with_required_motion_modes_for_task(
    validation_base_seed: u64,
    start_batch_idx: u64,
    temporal_task_mode: TemporalTaskMode,
) -> (Tensor, Tensor, u64) {
    for offset in 0..MIXED_MODE_SEARCH_LIMIT {
        let batch_idx = start_batch_idx + offset;
        let seed = validation_base_seed + batch_idx;
        let (x_t, x_t1) = make_temporal_batch_for_task(BATCH_SIZE, seed, temporal_task_mode);

        if batch_has_required_motion_mode_counts_for_task(&x_t, &x_t1, temporal_task_mode) {
            return (x_t, x_t1, seed);
        }
    }

    panic!(
        "did not find a required-mode validation batch for task {} within {} seeds from base {} and start batch {}",
        temporal_task_mode.as_str(),
        MIXED_MODE_SEARCH_LIMIT,
        validation_base_seed,
        start_batch_idx
    );
}

#[allow(dead_code)]
pub fn make_validation_batch_with_both_motion_modes_for_config(
    config: TemporalRunConfig,
    validation_base_seed: u64,
    start_batch_idx: u64,
) -> (Tensor, Tensor, u64) {
    make_validation_batch_with_both_motion_modes_for_task(
        validation_base_seed,
        start_batch_idx,
        config.temporal_task_mode,
    )
}

pub fn make_validation_batch_with_required_motion_modes_for_config(
    config: TemporalRunConfig,
    validation_base_seed: u64,
    start_batch_idx: u64,
) -> (Tensor, Tensor, u64) {
    make_validation_batch_with_required_motion_modes_for_task(
        validation_base_seed,
        start_batch_idx,
        config.temporal_task_mode,
    )
}

pub fn square_center_x(tensor: &Tensor, sample: usize) -> f32 {
    let mut weighted_sum = 0.0;
    let mut total_mass = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            let value = tensor.get(&[sample, 0, row, col]);
            weighted_sum += value * col as f32;
            total_mass += value;
        }
    }

    weighted_sum / total_mass
}

pub fn assert_temporal_contract(x_t: &Tensor, x_t1: &Tensor) {
    assert_eq!(
        x_t.shape,
        vec![BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    );
    assert_eq!(
        x_t1.shape,
        vec![BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    );

    for sample in 0..BATCH_SIZE {
        let center_x_t = square_center_x(x_t, sample);
        let center_x_t1 = square_center_x(x_t1, sample);
        let delta_x = center_x_t1 - center_x_t;
        let expected_motion_dx = signed_motion_dx_for_sample(x_t, x_t1, sample) as f32;
        let mass_t = total_mass(x_t, sample);
        let mass_t1 = total_mass(x_t1, sample);

        assert!(
            (delta_x - expected_motion_dx).abs() < 1e-6,
            "sample {} moved by {:.3}, expected {:.3}",
            sample,
            delta_x,
            expected_motion_dx
        );
        assert!(
            mass_t1 < mass_t,
            "sample {} mass did not follow the deterministic intensity rule: {:.3} -> {:.3}",
            sample,
            mass_t,
            mass_t1
        );
    }
}

pub fn print_batch_summary(name: &str, x_t: &Tensor, x_t1: &Tensor) {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);

    println!(
        "{} | shape {:?} | sample0 dx {} | slow {} | fast {} | center_x {:.3} -> {:.3} | mass {:.3} -> {:.3}",
        name,
        x_t.shape,
        motion_dx_for_sample(x_t, x_t1, 0),
        slow_count,
        fast_count,
        square_center_x(x_t, 0),
        square_center_x(x_t1, 0),
        total_mass(x_t, 0),
        total_mass(x_t1, 0)
    );
}

pub fn print_batch_summary_for_task(
    name: &str,
    temporal_task_mode: TemporalTaskMode,
    x_t: &Tensor,
    x_t1: &Tensor,
) {
    match temporal_task_mode {
        TemporalTaskMode::SignedVelocityTrail => {
            let (
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count,
            ) = signed_motion_mode_counts(x_t, x_t1);

            println!(
                "{} | shape {:?} | sample0 dx {} | -fast {} | -slow {} | +slow {} | +fast {} | center_x {:.3} -> {:.3} | mass {:.3} -> {:.3}",
                name,
                x_t.shape,
                signed_motion_dx_for_sample(x_t, x_t1, 0),
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count,
                square_center_x(x_t, 0),
                square_center_x(x_t1, 0),
                total_mass(x_t, 0),
                total_mass(x_t1, 0)
            );
        }
        TemporalTaskMode::RandomSpeed | TemporalTaskMode::VelocityTrail => {
            print_batch_summary(name, x_t, x_t1);
        }
    }
}

pub fn assert_required_motion_modes_for_task(
    temporal_task_mode: TemporalTaskMode,
    x_t: &Tensor,
    x_t1: &Tensor,
) {
    match temporal_task_mode {
        TemporalTaskMode::SignedVelocityTrail => {
            let (
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count,
            ) = signed_motion_mode_counts(x_t, x_t1);
            assert!(
                negative_fast_count >= MIN_SIGNED_MODE_COUNT
                    && negative_slow_count >= MIN_SIGNED_MODE_COUNT
                    && positive_slow_count >= MIN_SIGNED_MODE_COUNT
                    && positive_fast_count >= MIN_SIGNED_MODE_COUNT,
                "signed validation probe counts too small: -fast {} -slow {} +slow {} +fast {}, expected each >= {}",
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count,
                MIN_SIGNED_MODE_COUNT
            );
        }
        TemporalTaskMode::RandomSpeed | TemporalTaskMode::VelocityTrail => {
            let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);
            assert!(
                slow_count >= MIN_MIXED_MODE_COUNT,
                "mixed validation probe slow count too small: {} < {}",
                slow_count,
                MIN_MIXED_MODE_COUNT
            );
            assert!(
                fast_count >= MIN_MIXED_MODE_COUNT,
                "mixed validation probe fast count too small: {} < {}",
                fast_count,
                MIN_MIXED_MODE_COUNT
            );
        }
    }
}

pub fn print_motion_mode_summary_for_task(
    name: &str,
    seed: u64,
    temporal_task_mode: TemporalTaskMode,
    x_t: &Tensor,
    x_t1: &Tensor,
) {
    match temporal_task_mode {
        TemporalTaskMode::SignedVelocityTrail => {
            let (
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count,
            ) = signed_motion_mode_counts(x_t, x_t1);
            println!(
                "{} seed {} | -fast {} | -slow {} | +slow {} | +fast {}",
                name,
                seed,
                negative_fast_count,
                negative_slow_count,
                positive_slow_count,
                positive_fast_count
            );
        }
        TemporalTaskMode::RandomSpeed | TemporalTaskMode::VelocityTrail => {
            let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);
            println!(
                "{} seed {} | slow {} | fast {}",
                name, seed, slow_count, fast_count
            );
        }
    }
}

#[allow(dead_code)]
pub fn print_representation_stats(name: &str, stats: &RepresentationHealthStats) {
    println!(
        "{} health | mean_abs {:.6} | mean_std {:.6} | min_std {:.6} | offdiag_cov_abs {:.6} | offdiag_cov_max {:.6}",
        name,
        stats.mean_abs,
        stats.mean_std,
        stats.min_std,
        stats.mean_abs_offdiag_cov,
        stats.max_abs_offdiag_cov
    );
}

pub fn run_temporal_training_loop<TStep>(total_steps: usize, log_every: usize, mut step_fn: TStep)
where
    TStep: FnMut(usize, bool),
{
    assert!(total_steps > 0, "total_steps must be greater than 0");
    assert!(log_every > 0, "log_every must be greater than 0");

    for step in 1..=total_steps {
        let should_log = should_log_step(step, log_every);
        step_fn(step, should_log);
    }
}

#[allow(dead_code)]
pub fn print_temporal_train_val_metrics(step: usize, train_loss: f32, val_loss: f32) {
    println!(
        "step {:03} | train {:.6} | val {:.6}",
        step, train_loss, val_loss
    );
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn print_projected_temporal_train_val_metrics(
    step: usize,
    prediction_loss: f32,
    regularizer_loss: f32,
    total_loss: f32,
    target_projection_momentum: f32,
    target_projection_drift: f32,
    val_prediction_loss: f32,
    val_regularizer_loss: f32,
    val_total_loss: f32,
) {
    println!(
        "step {:03} | train pred {:.6} | reg {:.6} | total {:.6} | mom {:.4} drift {:.6}",
        step,
        prediction_loss,
        regularizer_loss,
        total_loss,
        target_projection_momentum,
        target_projection_drift
    );
    println!(
        "step {:03} | val pred {:.6} | reg {:.6} | val total {:.6}",
        step, val_prediction_loss, val_regularizer_loss, val_total_loss
    );
}

pub fn should_log_step(step: usize, log_every: usize) -> bool {
    assert!(log_every > 0, "log_every must be greater than 0");
    step == 1 || step.is_multiple_of(log_every)
}

const COMPACT_ENCODER_CHANNELS: usize = 6;
const COMPACT_ENCODER_STRONGER_CHANNELS: usize = 8;
const SIGNED_DIRECTION_ENCODER_CHANNELS: usize = 3;
const SIGNED_DIRECTION_KERNEL_HEIGHT: usize = 2;
const SIGNED_DIRECTION_KERNEL_WIDTH: usize = 5;

fn compact_encoder_channel_weights(row: usize, col: usize) -> [f32; COMPACT_ENCODER_CHANNELS] {
    let center = (IMAGE_SIZE as f32 - 1.0) / 2.0;
    let norm = (IMAGE_SIZE as f32 - 1.0).max(1.0);
    let x = (col as f32 - center) / norm;
    let y = (row as f32 - center) / norm;

    [
        1.0,   // total mass
        x,     // x centroid signal
        1.0,   // duplicate mass channel for compatibility with baseline
        y,     // y centroid signal
        x * x, // x dispersion signal
        y * y, // y dispersion signal
    ]
}

fn compact_encoder_channel_weights_stronger(
    row: usize,
    col: usize,
) -> [f32; COMPACT_ENCODER_STRONGER_CHANNELS] {
    let center = (IMAGE_SIZE as f32 - 1.0) / 2.0;
    let norm = (IMAGE_SIZE as f32 - 1.0).max(1.0);
    let x = (col as f32 - center) / norm;
    let y = (row as f32 - center) / norm;
    let xx = x * x;
    let yy = y * y;

    [
        1.0,     // total mass
        x,       // x centroid signal
        1.0,     // duplicate mass signal
        y,       // y centroid signal
        xx,      // x spread proxy
        yy,      // y spread proxy
        x * y,   // mixed/diagonal motion signal
        y.abs(), // directional magnitude
    ]
}

pub fn make_frozen_encoder() -> EmbeddingEncoder {
    let mut conv1_weights = Vec::with_capacity(3 * IMAGE_SIZE * IMAGE_SIZE);

    for _row in 0..IMAGE_SIZE {
        for _col in 0..IMAGE_SIZE {
            conv1_weights.push(1.0);
        }
    }

    for _row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            conv1_weights.push((col as f32 + 1.0) / IMAGE_SIZE as f32);
        }
    }

    for _row in 0..IMAGE_SIZE {
        for _col in 0..IMAGE_SIZE {
            conv1_weights.push(1.0);
        }
    }

    let conv1 = Conv2d::new(
        Tensor::new(conv1_weights, vec![3, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(vec![0.0, 0.0, -FAST_MOTION_MASS_THRESHOLD], vec![3]),
        1,
        0,
    );

    let conv2 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            vec![3, 3, 1, 1],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        1,
        0,
    );

    EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2))
}

pub fn make_compact_frozen_encoder() -> EmbeddingEncoder {
    let mut conv1_weights = Vec::with_capacity(COMPACT_ENCODER_CHANNELS * IMAGE_SIZE * IMAGE_SIZE);

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            conv1_weights.extend_from_slice(&compact_encoder_channel_weights(row, col));
        }
    }

    let conv1 = Conv2d::new(
        Tensor::new(
            conv1_weights,
            vec![COMPACT_ENCODER_CHANNELS, 1, IMAGE_SIZE, IMAGE_SIZE],
        ),
        Tensor::zeros(vec![COMPACT_ENCODER_CHANNELS]),
        1,
        0,
    );

    let conv2 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.10, 0.00, //
                0.0, 1.0, 0.0, 0.05, 0.00, 0.00, //
                0.0, 0.0, 1.0, 0.00, 0.00, 0.05,
            ],
            vec![3, COMPACT_ENCODER_CHANNELS, 1, 1],
        ),
        Tensor::new(vec![0.0, 0.0, -FAST_MOTION_MASS_THRESHOLD], vec![3]),
        1,
        0,
    );

    EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2))
}

pub fn make_compact_frozen_encoder_stronger() -> EmbeddingEncoder {
    let mut conv1_weights =
        Vec::with_capacity(COMPACT_ENCODER_STRONGER_CHANNELS * IMAGE_SIZE * IMAGE_SIZE);

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            conv1_weights.extend_from_slice(&compact_encoder_channel_weights_stronger(row, col));
        }
    }

    let conv1 = Conv2d::new(
        Tensor::new(
            conv1_weights,
            vec![COMPACT_ENCODER_STRONGER_CHANNELS, 1, IMAGE_SIZE, IMAGE_SIZE],
        ),
        Tensor::zeros(vec![COMPACT_ENCODER_STRONGER_CHANNELS]),
        1,
        0,
    );

    let conv2 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, 0.0, 1.0, 0.0, 0.10, 0.00, 0.02, -0.02, //
                0.0, 1.0, 0.0, 0.05, 0.00, 0.00, 0.01, 0.01, //
                0.0, 0.0, 1.0, 0.00, 0.00, 0.05, 0.01, -0.01, //
            ],
            vec![3, COMPACT_ENCODER_STRONGER_CHANNELS, 1, 1],
        ),
        Tensor::new(vec![0.0, 0.0, -FAST_MOTION_MASS_THRESHOLD], vec![3]),
        1,
        0,
    );

    EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2))
}

pub fn make_compact_frozen_encoder_signed_direction() -> EmbeddingEncoder {
    let mut conv1_weight = Tensor::zeros(vec![
        SIGNED_DIRECTION_ENCODER_CHANNELS,
        1,
        SIGNED_DIRECTION_KERNEL_HEIGHT,
        SIGNED_DIRECTION_KERNEL_WIDTH,
    ]);
    let odd_x = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let fast_gap = [0.5, 0.0, -1.0, 0.0, 0.5];

    for kh in 0..SIGNED_DIRECTION_KERNEL_HEIGHT {
        for kw in 0..SIGNED_DIRECTION_KERNEL_WIDTH {
            conv1_weight.set(&[0, 0, kh, kw], odd_x[kw]);
            conv1_weight.set(&[1, 0, kh, kw], -odd_x[kw]);
            conv1_weight.set(&[2, 0, kh, kw], fast_gap[kw]);
        }
    }

    let conv1 = Conv2d::new(
        conv1_weight,
        Tensor::zeros(vec![SIGNED_DIRECTION_ENCODER_CHANNELS]),
        1,
        0,
    );
    let conv2 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            vec![3, SIGNED_DIRECTION_ENCODER_CHANNELS, 1, 1],
        ),
        Tensor::zeros(vec![3]),
        1,
        0,
    );

    EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2))
}

fn draw_square(tensor: &mut Tensor, sample: usize, row: usize, col: usize, intensity: f32) {
    for dy in 0..SQUARE_SIZE {
        for dx in 0..SQUARE_SIZE {
            tensor.set(&[sample, 0, row + dy, col + dx], intensity);
        }
    }
}

fn add_square(tensor: &mut Tensor, sample: usize, row: usize, col: usize, intensity: f32) {
    for dy in 0..SQUARE_SIZE {
        for dx in 0..SQUARE_SIZE {
            let index = [sample, 0, row + dy, col + dx];
            let next_value = tensor.get(&index) + intensity;
            tensor.set(&index, next_value);
        }
    }
}

fn squares_overlap(row_a: usize, col_a: usize, row_b: usize, col_b: usize) -> bool {
    let row_a_end = row_a + SQUARE_SIZE;
    let col_a_end = col_a + SQUARE_SIZE;
    let row_b_end = row_b + SQUARE_SIZE;
    let col_b_end = col_b + SQUARE_SIZE;

    row_a < row_b_end && row_b < row_a_end && col_a < col_b_end && col_b < col_a_end
}

pub fn total_mass(tensor: &Tensor, sample: usize) -> f32 {
    let mut total = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            total += tensor.get(&[sample, 0, row, col]);
        }
    }

    total
}

#[cfg(test)]
#[allow(dead_code)]
pub fn active_cell_count(tensor: &Tensor, sample: usize) -> usize {
    let mut count = 0usize;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            if tensor.get(&[sample, 0, row, col]).abs() > 1e-6 {
                count += 1;
            }
        }
    }

    count
}

#[cfg(test)]
#[allow(dead_code)]
pub fn assert_square_footprint_and_decay_invariants(
    x_t: &Tensor,
    x_t1: &Tensor,
    seed: u64,
) -> (usize, usize) {
    let mut single_square_samples = 0usize;
    let mut double_square_samples = 0usize;

    for sample in 0..BATCH_SIZE {
        let cells_t = active_cell_count(x_t, sample);
        let cells_t1 = active_cell_count(x_t1, sample);

        match cells_t {
            4 => single_square_samples += 1,
            8 => double_square_samples += 1,
            cells => panic!(
                "unexpected non-zero footprint at seed {} sample {}: {}",
                seed, sample, cells
            ),
        }

        assert!(
            cells_t == cells_t1,
            "non-zero footprint changed across temporal step at seed {} sample {}: {} -> {}",
            seed,
            sample,
            cells_t,
            cells_t1
        );

        let mass_t = total_mass(x_t, sample);
        let mass_t1 = total_mass(x_t1, sample);
        assert!(
            mass_t1 < mass_t,
            "sample {} mass did not decay at seed {}: {:.6} -> {:.6}",
            sample,
            seed,
            mass_t,
            mass_t1
        );
    }

    (single_square_samples, double_square_samples)
}

#[cfg(test)]
#[allow(dead_code)]
pub fn assert_seed_range_has_single_and_double_square_batch_examples(
    seed_count: u64,
    mut make_batch: impl FnMut(u64) -> (Tensor, Tensor),
) -> (usize, usize) {
    let mut saw_single_square_batches = 0usize;
    let mut saw_double_square_batches = 0usize;

    for seed in 0..seed_count {
        let (x_t, x_t1) = make_batch(seed);
        let (single_count, double_count) =
            assert_square_footprint_and_decay_invariants(&x_t, &x_t1, seed);

        saw_single_square_batches += usize::from(single_count > 0);
        saw_double_square_batches += usize::from(double_count > 0);
    }

    assert!(
        saw_single_square_batches > 0,
        "never saw single-square sample in {} seeds",
        seed_count
    );
    assert!(
        saw_double_square_batches > 0,
        "never saw double-square sample in {} seeds",
        seed_count
    );

    (saw_single_square_batches, saw_double_square_batches)
}

#[cfg(test)]
#[allow(dead_code)]
pub fn assert_seed_range_has_both_motion_modes(
    seed_count: u64,
    mut make_batch: impl FnMut(u64) -> (Tensor, Tensor),
) -> (bool, bool) {
    let mut saw_slow_motion = false;
    let mut saw_fast_motion = false;

    for seed in 0..seed_count {
        let (x_t, x_t1) = make_batch(seed);

        for sample in 0..BATCH_SIZE {
            match motion_dx_for_sample(&x_t, &x_t1, sample) {
                SLOW_MOTION_DX => saw_slow_motion = true,
                FAST_MOTION_DX => saw_fast_motion = true,
                dx => panic!(
                    "unexpected motion dx {} in seed {} sample {}",
                    dx, seed, sample
                ),
            }
        }
    }

    assert!(
        saw_slow_motion,
        "never observed slow motion in {} seeds",
        seed_count
    );
    assert!(
        saw_fast_motion,
        "never observed fast motion in {} seeds",
        seed_count
    );

    (saw_slow_motion, saw_fast_motion)
}
