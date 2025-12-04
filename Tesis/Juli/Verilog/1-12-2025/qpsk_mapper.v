`timescale 1ns / 1ps

module qpsk_mapper #(
    parameter WIDTH = 20 // <--- CAMBIO A 20 BITS
)(
    input  wire             bit_in,
    output wire signed [WIDTH-1:0] sym_out
);

    // Constante: 0.7071 * 2^10 = 724
    // Formato Q9.10 (1 signo, 9 enteros, 10 fracción)
    localparam signed [WIDTH-1:0] VAL_POS = 20'd724;
    localparam signed [WIDTH-1:0] VAL_NEG = -20'd724;

    assign sym_out = (bit_in == 1'b0) ? VAL_POS : VAL_NEG;

endmodule