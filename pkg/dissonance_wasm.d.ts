/* tslint:disable */
/* eslint-disable */
/**
*/
export function greet(): void;
/**
* @param {Float64Array} freqs
* @returns {number}
*/
export function calculateDissonance(freqs: Float64Array): number;
/**
* @param {Array<any>} matrix
* @returns {Int32Array}
*/
export function dissonanceMatrix(matrix: Array<any>): Int32Array;
/**
* @param {Float64Array} freqs
* @returns {number}
*/
export function findOffender(freqs: Float64Array): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly greet: () => void;
  readonly calculateDissonance: (a: number, b: number) => number;
  readonly dissonanceMatrix: (a: number, b: number) => void;
  readonly findOffender: (a: number, b: number) => number;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
