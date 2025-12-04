module tx_top (
    input  wire clk,
    input  wire reset,
    output wire signed [15:0] sI_out, // Recomendado: 16 bits para entrar a la FFT
    output wire signed [15:0] sQ_out
);

    // Señales internas
    wire bI, bQ;

    // 1. PRBS I (Semilla 0x17F)
    // NOTA: 'seed' es un puerto, no un parámetro, según tu prbs9.v
    prbs9 prbs_i (
        .clk (clk),
        .rst (reset),
        .en  (1'b1),      // <--- IMPORTANTE: Habilitar el PRBS
        .seed(9'h17F),    // <--- Conectado al puerto 'seed'
        .bit (bI)
    );

    // 2. PRBS Q (Semilla 0x11D)
    prbs9 prbs_q (
        .clk (clk),
        .rst (reset),
        .en  (1'b1),      // <--- IMPORTANTE
        .seed(9'h11D),
        .bit (bQ)
    );

    // 3. Mapeador QPSK
    // Asegúrate de que qpsk_mapper.v tenga puertos que coincidan con estos nombres
    qpsk_mapper mapper_qpsk (
        .bit_I(bI),       // Nombres sugeridos en el paso anterior
        .bit_Q(bQ),
        .sym_I(sI_out),
        .sym_Q(sQ_out)
    );

endmodule