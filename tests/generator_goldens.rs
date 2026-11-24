// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden topology digests for the deterministic generators.
//!
//! Values were recorded on `main` before the generators gained
//! `*_with_model` siblings. A change to any of them changes generated graphs:
//! record it in the CHANGELOG before updating the golden.

use synaptic_wiring::topology::{
    generate_layered, generate_random, generate_scale_free, generate_small_world,
};

const PREFIX: &str = "synaptic-wiring.topology.digest.v1:sha256:";

/// `(max_delay, [random, small_world, scale_free, layered])`.
const GOLDENS: [(u16, [&str; 4]); 3] = [
    (
        0,
        [
            "e79242c10503a350b5f8deb71584e981304d583b5980dc5dfd455655232db146",
            "5845e8c3e551d512d9a09d0465665279a8ad3b13711f152fabc7af5957b9dee3",
            "2273737fce4cd33e3e43cd8c7da8f2d9191075f00ed48f7d589bc56021b1f488",
            "b8f62b12af0d8163a0267e9c67dd49364e04c4917190045996258ffaadf911d0",
        ],
    ),
    (
        1,
        [
            "f7a18b787e1570a1e8e3462a9dd4f0142a8255d1d38266523c49f00f7baa4ced",
            "cfd9e80ccf0387e9ad54e4b9eabc4addea53ad4445ee240040b44895412dd009",
            "32d9937511f82fd0c38ce95db73c99b54c6c3d3c86b8efe0c8b7cf50eda0c9c9",
            "cc86a0407faebfceafec1f7457c8811eb2c7359c1b487511d672877e122963da",
        ],
    ),
    (
        7,
        [
            "1a3c94ee1f0eab42ee5d72a719c741e7cb7faa90c8e5175aca29f9bb821cebdc",
            "d475ccafdb86621c6e596c52f9006778f00760903985a9128bf3013ca181d6fb",
            "c598c463b9d23f4861684942a234c31895f83c95e1e31ff78e8262fd2193b901",
            "33fab74d609372fdecc843d247578dd81462e5f6e80eea0868e6f4cf7710890b",
        ],
    ),
];

#[test]
fn generator_digests_match_goldens() {
    for (max_delay, [random, small_world, scale_free, layered]) in GOLDENS {
        let actual = [
            generate_random(48, 0.15, max_delay, 0.2),
            generate_small_world(48, 6, 0.3, max_delay, 0.2),
            generate_scale_free(48, 4, 3, max_delay, 0.2),
            generate_layered(&[16, 20, 12], 0.4, max_delay, 0.2),
        ];
        for (graph, golden) in actual
            .into_iter()
            .zip([random, small_world, scale_free, layered])
        {
            let digest = graph.unwrap().topology_digest().to_string();
            assert_eq!(digest, format!("{PREFIX}{golden}"), "max_delay={max_delay}");
        }
    }
}
