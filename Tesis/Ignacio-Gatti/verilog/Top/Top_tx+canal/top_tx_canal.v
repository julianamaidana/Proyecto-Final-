`timescale 1ns/1ps

module top #(
    parameter DWIDTH      = 9,
    parameter SNR_WIDTH   = 11
)(
    input  wire                        clk,
    input  wire                        rst,         
    input  wire signed [SNR_WIDTH-1:0] sigma_scale, // Control de nivel de ruido
    input wire i_backpressure,         // Control de backpressure
    
    // Salida final del sistema (Señal transmitida + Canal + Ruido)
    output wire signed [DWIDTH-1:0]    rx_I,
    output wire signed [DWIDTH-1:0]    rx_Q
);

    // ============================================================
    // CABLES INTERNOS (Interconexión TX -> Canal)
    // ============================================================
    wire signed [DWIDTH-1:0] tx_I_internal;
    wire signed [DWIDTH-1:0] tx_Q_internal;


    wire system_enable = !i_backpressure;

    // ============================================================
    // 1. INSTANCIA DEL TRANSMISOR (TX)
    // ============================================================
    // Genera la secuencia PRBS y mapea a QPSK
    tx_top u_tx_top (
        .clk    (clk),
        .reset  (rst),    
        .i_enable(system_enable), // Control de backpressure        
        .sI_out (tx_I_internal),   // Salida I del TX
        .sQ_out (tx_Q_internal)    // Salida Q del TX
    );

    // ============================================================
    // 2. INSTANCIA DEL CANAL (CHANNEL)
    // ============================================================
    // Aplica filtros FIR (ISI) y agrega ruido Gaussiano
    channel_with_noise #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_channel (
        .clk         (clk),
        .rst         (rst),
        .i_enable    (system_enable),
        .In_I        (tx_I_internal), // Conectamos la salida del TX aquí
        .In_Q        (tx_Q_internal), // Conectamos la salida del TX aquí
        .sigma_scale (sigma_scale),
        .Out_I       (rx_I),          // Salida final hacia el mundo exterior
        .Out_Q       (rx_Q)
    );

endmodule