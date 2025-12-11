 `timescale 1ns / 1ps

module tx_top #(
    parameter DATA_WIDTH = 20 // <--- CAMBIO A 20 BITS
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       en,
    output wire signed [DATA_WIDTH-1:0] tx_i,
    output wire signed [DATA_WIDTH-1:0] tx_q
);
    // --- Configuración de Semillas (Según tu requerimiento) ---
    localparam [8:0] SEED_I = 9'h1AA;
    localparam [8:0] SEED_Q = 9'h1FE;

    // --- Cables internos para conectar PRBS con Mapper ---
    wire bit_internal_i;
    wire bit_internal_q;

    // ============================================================
    // RAMA I (IN-PHASE)
    // ============================================================
    
    // 1. Generador PRBS para I
    prbs9 #(
        .SEED(SEED_I)
    ) u_prbs_i (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .bit_out(bit_internal_i)
    );

    // 2. Mapper para I
    qpsk_mapper #(
        .WIDTH(DATA_WIDTH)
    ) u_mapper_i (
        .bit_in(bit_internal_i),
        .sym_out(tx_i)
    );

    // ============================================================
    // RAMA Q (QUADRATURE)
    // ============================================================

    // 1. Generador PRBS para Q
    prbs9 #(
        .SEED(SEED_Q)
    ) u_prbs_q (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .bit_out(bit_internal_q)
    );

    // 2. Mapper para Q
    qpsk_mapper #(
        .WIDTH(DATA_WIDTH)
    ) u_mapper_q (
        .bit_in(bit_internal_q),
        .sym_out(tx_q)
    );

endmodule