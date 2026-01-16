# wasm-tools workload extension

This package is an extension for the wasm-tools dotnet workload. It ships LLVM
utilities that are missing, so tooling that targets
browser-wasm can work out of the box.

Included utilities:
- llc: LLVM static compiler (IR to target assembly or object code).
- opt: LLVM optimizer and IR transformation pipeline driver.
- llvm-objcopy: object file copier/stripper for section and symbol edits.
- llvm-dis: LLVM bitcode disassembler (bitcode to textual IR).
- llvm-as: LLVM assembler (textual IR to bitcode).
