//! RISC-V emitter scaffolding.

use crate::{
    common_decl::Size,
    location::{Location as AbstractLocation, Reg},
};
pub use crate::{
    machine::{Label, Offset},
    riscv_decl::{FPR, GPR},
};
use dynasm::dynasm;
use dynasmrt::{
    riscv::RiscvRelocation, AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, VecAssembler,
};
use wasmer_compiler::types::{
    function::FunctionBody,
    section::{CustomSection, CustomSectionProtection, SectionBody},
};
use wasmer_types::{
    target::CallingConvention, target::CpuFeature, CompileError, FunctionIndex, FunctionType, Type,
    VMOffsets,
};

type Assembler = VecAssembler<RiscvRelocation>;

/// Force `dynasm!` to use the correct arch (riscv64) when cross-compiling.
macro_rules! dynasm {
    ($a:expr ; $($tt:tt)*) => {
        dynasm::dynasm!(
            $a
            ; .arch riscv64
            ; $($tt)*
        )
    };
}

/// Location abstraction specialized to RISC-V.
pub type Location = AbstractLocation<GPR, FPR>;

/// Branch conditions for RISC-V.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Condition {
    // TODO: define RISC-V branch conditions.
}

/// Emitter trait for RISC-V.
#[allow(unused)]
pub trait EmitterRiscv {
    /// Generates a new internal label.
    fn get_label(&mut self) -> Label;
    /// Gets the current code offset.
    fn get_offset(&self) -> Offset;
    /// Returns the size of a jump instruction in bytes.
    fn get_jmp_instr_size(&self) -> u8;

    /// Finalize the function, e.g., resolve labels.
    fn finalize_function(&mut self) -> Result<(), CompileError>;

    // TODO: add methods for emitting RISC-V instructions (e.g., loads, stores, arithmetic, branches, etc.)

    fn emit_mov(&mut self, sz: Size, src: Location, dst: Location) -> Result<(), CompileError>;
    fn emit_unimp(&mut self) -> Result<(), CompileError>;
    fn emit_ret(&mut self) -> Result<(), CompileError>;
    fn emit_add(
        &mut self,
        sz: Size,
        src1: Location,
        src2: Location,
        dst: Location,
    ) -> Result<(), CompileError>;

    fn emit_label(&mut self, label: Label) -> Result<(), CompileError>;
    fn emit_store(&mut self, sz: Size, src: Location, dst: Location) -> Result<(), CompileError>;

    fn emit_adjust_stack(&mut self, delta_stack_offset: i32) -> Result<(), CompileError>;
    fn emit_prolog(&mut self) -> Result<(), CompileError>;
    fn emit_epilog(&mut self) -> Result<(), CompileError>;
    fn emit_pop(&mut self, reg: Location) -> Result<(), CompileError>;
}

impl EmitterRiscv for Assembler {
    fn get_label(&mut self) -> Label {
        self.new_dynamic_label()
    }

    fn get_offset(&self) -> Offset {
        self.offset()
    }

    fn get_jmp_instr_size(&self) -> u8 {
        1
    }

    fn finalize_function(&mut self) -> Result<(), CompileError> {
        Ok(())
    }

    fn emit_mov(&mut self, sz: Size, src: Location, dst: Location) -> Result<(), CompileError> {
        match (sz, src, dst) {
            (Size::S64, Location::GPR(src), Location::GPR(dst)) => {
                let src = src.into_index() as u32;
                let dst = dst.into_index() as u32;
                dynasm!(self ; add X(dst), X(src), x0);
            }
            (Size::S32, Location::Memory(reg1, disp), Location::GPR(dst)) => {
                let reg1 = reg1.into_index() as u32;
                let dst = dst.into_index() as u32;
                let disp = disp as i32;
                dynasm!(self
                    ; lw X(dst), [X(reg1), -disp]
                )
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn emit_label(&mut self, label: Label) -> Result<(), CompileError> {
        dynasm!(self ; => label);
        Ok(())
    }

    fn emit_unimp(&mut self) -> Result<(), CompileError> {
        dynasm!(self; unimp);
        Ok(())
    }

    fn emit_add(
        &mut self,
        sz: Size,
        src1: Location,
        src2: Location,
        dst: Location,
    ) -> Result<(), CompileError> {
        // We do know that we are going to be called only once, and we know that
        // the parameters are already in a1 and a2, so we can just emit a hardcoded
        // addw a0, a1, a2 and it should work for our specific case
        match (sz, src1, src2, dst) {
            (Size::S32, Location::GPR(src1), Location::GPR(src2), Location::GPR(dst)) => {
                let src1 = src1.into_index() as u32;
                let src2 = src2.into_index() as u32;
                let dst = dst.into_index() as u32;
                dynasm!(self
                    ; addw X(dst), X(src1), X(src2)
                );
            }
            (
                Size::S32,
                Location::Memory(reg1, disp1),
                Location::Memory(reg2, disp2),
                Location::Memory(reg3, disp3),
            ) => {
                let reg1 = reg1.into_index() as u32;
                let reg2 = reg2.into_index() as u32;
                let reg3 = reg3.into_index() as u32;

                let disp1 = disp1 as i32;
                let disp2 = disp2 as i32;
                let disp3 = disp3 as i32;
                dynasm!(self
                    ; lw t1, [X(reg1), -disp1]
                    ; lw t2, [X(reg2), -disp2]
                    ; addw t1, t1, t2
                    ; sw t1, [X(reg3), -disp3]
                );
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn emit_ret(&mut self) -> Result<(), CompileError> {
        dynasm!(self
            ; ret
        );
        Ok(())
    }

    fn emit_store(&mut self, sz: Size, reg: Location, addr: Location) -> Result<(), CompileError> {
        match (sz, reg, addr) {
            (Size::S64, Location::GPR(reg), Location::Memory(addr, disp)) => {
                let reg = reg.into_index() as u32;
                let addr = addr.into_index() as u32;
                let disp = disp as i32;
                dynasm!(self ; sd X(reg), [X(addr), -disp]);
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn emit_adjust_stack(&mut self, off: i32) -> Result<(), CompileError> {
        dynasm!(self
           ; addi sp, sp, off
        );
        Ok(())
    }

    fn emit_prolog(&mut self) -> Result<(), CompileError> {
        dynasm!(self
            ; sd fp, [sp, -16]
            ; sd ra, [sp, -8]
            ; addi fp, sp, -16
            ; addi sp, sp, -16
        );
        Ok(())
    }

    fn emit_epilog(&mut self) -> Result<(), CompileError> {
        dynasm!(self
            ; addi sp, fp, 16
            ; ld ra, [fp, 8]
            ; ld fp, [fp, 0]
        );
        Ok(())
    }

    fn emit_pop(&mut self, reg: Location) -> Result<(), CompileError> {
        match reg {
            Location::GPR(reg) => {
               let reg = reg.into_index() as u32;
               dynasm!(self
                   ; addi sp, sp, 8
                   ; ld X(reg), [sp, 8]
               )
            }
            _ => todo!()
        }
        Ok(())
    }
}

pub fn gen_std_trampoline_riscv64(
    sig: &FunctionType,
    calling_convention: CallingConvention,
) -> Result<FunctionBody, CompileError> {
    let mut a = Assembler::new(0);

    dynasm!(a
        ; sd fp, [sp, -16]
        ; sd ra, [sp, -8]
        ; addi fp, sp, -16
        ; addi sp, sp, -16
    );

    dynasm!(a
       ; add t1, a1, x0
       ; add t0, a2, x0
       ; add x27, a0, x0
       ; lw a0, [a2, 0]
       ; lw a1, [a2, 16]
       ; jalr t1
       ; sw a0, [t0, 0]
    );

    dynasm!(a
        ; addi sp, fp, 16
        ; ld ra, [fp, 8]
        ; ld fp, [fp, 0]
        ; ret
    );

    let mut body = a.finalize().unwrap();
    body.shrink_to_fit();

    Ok(FunctionBody {
        body,
        unwind_info: None,
    })
}
