use jepra_core::{Tensor, representation_health_stats};

#[test]
fn representation_stats_reports_per_dimension_spread() {
    let latents = Tensor::new(
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ],
        vec![3, 2],
    );

    let stats = representation_health_stats(&latents);

    assert_eq!(stats.per_dim_std.len(), 2);
    assert!((stats.mean_abs - 3.5).abs() < 1e-6);
    assert!((stats.mean_std - 1.6329932).abs() < 1e-6);
    assert!((stats.min_std - 1.6329932).abs() < 1e-6);
    assert!((stats.mean_abs_offdiag_cov - 2.6666667).abs() < 1e-6);
    assert!((stats.max_abs_offdiag_cov - 2.6666667).abs() < 1e-6);
}

#[test]
fn representation_stats_flags_collapsed_features() {
    let latents = Tensor::new(vec![1.0, 7.0, 1.0, 8.0, 1.0, 9.0], vec![3, 2]);

    let stats = representation_health_stats(&latents);

    assert_eq!(stats.min_std, 0.0);
    assert_eq!(stats.mean_abs_offdiag_cov, 0.0);
    assert_eq!(stats.max_abs_offdiag_cov, 0.0);
}
