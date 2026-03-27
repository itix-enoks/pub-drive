/* change the bit fields and instruction fields */
def CUSTOM_AES32ESMI : Instruction {
    bits<32> Inst;
    bits<32> SoftFail = 0;
    bits<2> bs;
    bits<5> rs2;
    bits<5> rs1;
    bits<5> rd;
    let Namespace = "RISCV";
    let Size = 4;
    let hasSideEffects = 0;
    let mayLoad = 0;
    let mayStore = 0;
    let Inst{31-30} = bs;
    let Inst{29-25} = 0b10100; /*funct5*/  
    let Inst{24-20} = rs2;
    let Inst{19-15} = rs1;
    let Inst{14-12} = 0b000; /*funct3*/
    let Inst{11-7} = rd;
    let Inst{6-0} = 0b0110011; /*opcode*/
    dag OutOperandList = (outs GPR:$rd);
    dag InOperandList = (ins GPR:$rs1, GPR:$rs2, uimm2:$bs);
    let AsmString = "aes32esmi_custom\t$rd, $rs1, $rs2, $bs";
}

def : Pat<(int_riscv_aes32esmi GPR:$rs1, GPR:$rs2, uimm2:$bs), (CUSTOM_AES32ESMI $rs1, $rs2, $bs)>;
