// Hand-written type declarations for dengo-solver.js -- see that file
// for the full documentation of behavior/gotchas; kept in sync by hand
// (this project deliberately isn't running a TypeScript build step for
// wasm/, see NOTES.md for why -- this file is the whole of that
// tradeoff's cost, and it's small).

/** The subset of a dengo-generated Emscripten module's surface this
 * class relies on -- not the module's full type, just what's actually
 * used, so any real dengo wasm module (this project's or your own)
 * satisfies it without extra work. */
export interface DengoWasmModule {
  HEAPF64: Float64Array;
  cwrap(
    ident: string,
    returnType: "number" | "string" | null,
    argTypes?: ("number" | "string")[]
  ): (...args: any[]) => any;
}

/** A dengo-generated Emscripten module factory -- the page-global
 * `DengoModule` from a `<script src="dengo_wasm.js">`, or
 * `require("./dengo_wasm.js")`/`import("./dengo_wasm.mjs")` in Node or
 * a bundler; the same compiled output supports all three. */
export type DengoModuleFactory = (options?: unknown) => Promise<DengoWasmModule>;

export class DengoSolver {
  constructor(mod: DengoWasmModule);
  static create(moduleFactory: DengoModuleFactory): Promise<DengoSolver>;

  /** Every species this network's state vector holds (including `ge`),
   * in the exact order the state buffer uses. Frozen. */
  readonly speciesNames: readonly string[];
  /** speciesNames.length. */
  readonly nspecies: number;

  readState(): Record<string, number>;
  writeState(values: Record<string, number>): void;
  get(name: string): number;
  set(name: string, value: number): void;
  bulkUpdate(
    fn: (heap: Float64Array, ptr: number, index: Readonly<Record<string, number>>) => void
  ): void;
  rhs(): Record<string, number>;
  /** Kelvin; see dengo-solver.js's module doc comment on staleness. */
  temperature(): number;
  step(dtf: number, niter?: number, reltol?: number): boolean;
}
