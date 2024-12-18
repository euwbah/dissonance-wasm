pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub fn cents_to_hz(base: f64, cents: f64) -> f64 {
    base * 2.0f64.powf(cents / 1200.0)
}

pub fn hz_to_cents(base: f64, hz: f64) -> f64 {
    1200.0 * (hz / base).log2()
}

#[cfg(test)]
mod tests {
}
