`timescale 1ns/1ps

module qpsk_mapper (
    // Entrada: Bits crudos del PRBS
    input  wire bit_I,
    input  wire bit_Q,
    
    // Salida: Símbolos QPSK en formato Q9.7 (16 bits signed)
    // +90 (0x005A) o -90 (0xFFA6)
    output wire signed [15:0] sym_I,
    output wire signed [15:0] sym_Q
);

    // Definición de la amplitud según tu generate_test_vectors.py
    // Valor 90 decimal. En Hex: 16'h005A
    localparam signed [15:0] QPSK_VAL_POS = 16'sd90;
    
    // Valor -90 decimal. En complemento a 2 (16 bits): 16'hFFA6
    localparam signed [15:0] QPSK_VAL_NEG = -16'sd90;

    // Lógica de Mapeo:
    // Si bit es 0 -> Positivo (+90)
    // Si bit es 1 -> Negativo (-90)
    assign sym_I = (bit_I == 1'b0) ? QPSK_VAL_POS : QPSK_VAL_NEG;
    assign sym_Q = (bit_Q == 1'b0) ? QPSK_VAL_POS : QPSK_VAL_NEG;

endmodule