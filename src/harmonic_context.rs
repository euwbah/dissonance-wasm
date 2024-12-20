//! This module's job is to:
//!
//! - Figure out how to decide which notes to keep in the harmonic context when too many notes are
//!   inside.
//!
//!     * Do we drop the note first, then add (faster, less accurate)? Or do we add first then drop
//!       (very slow, more accurate)?
//!
//! - Given a list of possible detemperaments of a new note, evaluate the best possible
//!   detemperament of a note.
//!
//! - Update the perception of tonicities over time.
//!
//! - Enforce tonicity constraints on notes that are still currently being played.
//!
//!
//! From past experiments, here's the strategy.
//!
//! - To find which candidate detemperament to add, run graph_diss with max 2 attempts per
//!   traversal, ignore tonicity and only consider dissonance. Choose the least dissonant option.
//!
//! - After adding a note, or in the update step, we evaluate tonicity using 3 attemps per
//!   traversal, with max 3000 completed traversals.
//!
//! - When a new note is to be added but the harmonic context is full, we first drop the note with:
//!
//!     1. The highest dissonance contribution, evaluated by computing the dissonances of all (N-1)
//!        subsets (using the fast 2 attempts per traversal setting) and finding the omitted note
//!        that reduces dissonance by the most.
//!
//!     2. The lowest tonicity.
//!
//!   TODO: figure out which one is better.
