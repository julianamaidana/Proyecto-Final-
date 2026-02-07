`timescale 1ns/1ps

module top #(
    parameter DWIDTH      = 9,
    parameter SNR_WIDTH   = 11
)(
    input  wire                        i_clk,         // Reloj del sistema
    input  wire                        i_rst,         // Reset (activo en alto) 
    input  wire signed [SNR_WIDTH-1:0] i_sigma_scale, // Control de nivel de ruido
    input  wire                        i_tx_en,       // Habilitación (freno desde el buffer) 
    
    // Salidas hacia el buffer de Overlap
    output wire signed [DWIDTH-1:0]    o_rx_I,        // Señal de salida I 
    output wire signed [DWIDTH-1:0]    o_rx_Q,        // Señal de salida Q 
    output wire                        o_rx_valid     // Señal de validez (handshake forward)
);

    // ============================================================
    // CABLES INTERNOS (Interconexión TX -> Canal)
    // ============================================================
    wire signed [DWIDTH-1:0] tx_I_internal; 
    wire signed [DWIDTH-1:0] tx_Q_internal; 

    // ============================================================
    // 1. INSTANCIA DEL TRANSMISOR (TX)
    // ============================================================
    tx_top u_tx_top (
        .clk    (i_clk),    // Conexión al nuevo nombre de puerto 
        .reset  (i_rst),    // Conexión al nuevo nombre de puerto 
        .in_en  (i_tx_en),  // PRBS avanza solo si el buffer lo permite
        .sI_out (tx_I_internal), 
        .sQ_out (tx_Q_internal)  
    );

    // ============================================================
    // 2. INSTANCIA DEL CANAL (CHANNEL)
    // ============================================================
    channel_with_noise #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_channel (
        .clk         (i_clk),         // [cite: 72]
        .rst         (i_rst),         // [cite: 72]
        .In_I        (tx_I_internal), // 
        .In_Q        (tx_Q_internal), // 
        .sigma_scale (i_sigma_scale), // 
        .Out_I       (o_rx_I),        // 
        .Out_Q       (o_rx_Q)         // 
    );

    // ============================================================
    // 3. LÓGICA DE VALIDEZ
    // ============================================================
    // El dato es válido si el sistema estuvo habilitado para procesar.
    assign o_rx_valid = i_tx_en;

endmodule