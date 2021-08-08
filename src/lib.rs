mod utils;

use wasm_bindgen::prelude::*;
use ndarray::Array1;
use std::cmp::max;
use js_sys::Array;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, dissonance-wasm!");
}

const A: f64 = 3.5;
const B: f64 = 5.75;
const D_MAX: f64 = 0.24;
const S_1: f64 = 0.0207;
const S_2: f64 = 18.96;

#[wasm_bindgen(js_name=calculateDissonance)]
pub fn calculate_dissonance(freqs: &[f64]) -> f64 {
    let mut pairs = Vec::new();
    for f in freqs {
        for h in 1..=16 {
            pairs.push((*f * h as f64, 0.88f64.powi(h - 1)));
        }
    }

    pairs.sort_by(|(f1, _), (f2, _)| f1.partial_cmp(f2).unwrap());

    let mut dissonance = 0f64;

    for i in 0..(pairs.len() - 1) {
        for j in (i + 1)..(pairs.len()) {
            let (f1, a1) = pairs[i];
            let (f2, a2) = pairs[j];

            let spl_amplitude_of_interference = a1.max(a2);
            let s = D_MAX / (S_1 * f1 + S_2);
            let x = s * (f2 - f1); // f2 > f1 is a must
            let d = (-A * x).exp() - (-B * x).exp();
            dissonance += spl_amplitude_of_interference * d;
        }
    }

    return dissonance;
}

#[wasm_bindgen(js_name=dissonanceMatrix)]
pub fn dissonance_matrix(matrix: Array) -> Vec<i32> {
    let mut min_dissonance = 99999f64;
    let mut p_cand = -1i32;
    let mut r_cand = -1i32;
    for (i, pitch_candidate) in matrix.iter().enumerate() {
        let pitch_candidate: Array = Array::from(&pitch_candidate);
        for (j, ratio_candidate) in pitch_candidate.iter().enumerate() {
            let ratio_candidate: Vec<JsValue> = Array::to_vec(&Array::from(&ratio_candidate));
            let freqs = ratio_candidate.iter().map(|x| x.as_f64().unwrap()).collect::<Vec<f64>>();
            let diss = calculate_dissonance(freqs.as_slice());
            if diss < min_dissonance {
                min_dissonance = diss;
                p_cand = i as i32;
                r_cand = j as i32;
            }
        }
    }

    return vec![p_cand, r_cand];
}

#[wasm_bindgen(js_name=findOffender)]
pub fn find_offender(freqs: &[f64]) -> usize {
    let mut min_dissonance = 99999f64;
    let mut offender = 0;

    for i in 0..freqs.len()-1 {
        let (left, right) = freqs.split_at(i);
        let minus_one = [left, &right[1..]].concat();
        let diss = calculate_dissonance(&minus_one);
        if diss < min_dissonance {
            min_dissonance = diss;
            offender = i;
        }
    }

    return offender;
}