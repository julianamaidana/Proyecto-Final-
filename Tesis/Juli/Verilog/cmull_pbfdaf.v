`timescale 1ns/1ps
`default_nettype none

// ============================================================
// cmul_pbfdaf
//
// Propósito:
//   Calcula Y[k] = W0[k]·X0[k] + W1[k]·X1[k] en streaming.
//   Es el bloque de filtrado frecuencial del PBFDAF con K=2.
//
// Interfaz de datos:
//   - Entrada: streaming bin a bin (k=0..31), 1 ciclo por bin.
//   - X0/X1 vienen del history_buffer.
//   - W0/W1 se leen de la RAM interna, sincronizados con k.
//   - Salida: Y[k] con 1 ciclo de latencia.
//
// Puerto de escritura de pesos (para LMS):
//   - i_we    : write enable
//   - i_wk    : índice k (0..NFFT-1)
//   - i_wsel  : 0=W0, 1=W1
//   - i_W_re  : parte real del peso
//   - i_W_im  : parte imaginaria del peso
//   La escritura puede ocurrir en cualquier ciclo.
//   Si coincide con una lectura del mismo bin, la escritura
//   toma efecto en el ciclo siguiente (read-before-write).
//
// Inicialización de pesos:
//   W0[k] = 1.0 = 2^NBF_W para todos k  (identidad)
//   W1[k] = 0              para todos k
//
// Latencia de salida: 1 ciclo (registro de salida).
// ============================================================

module cmul_pbfdaf #(
    parameter integer NB_W  = 17,
    parameter integer NBF_W = 10,
    parameter integer NFFT  = 32
)(
    input  wire                      clk,
    input  wire                      rst,

    // --- Entrada streaming: del history_buffer ---
    input  wire                      i_valid,
    input  wire                      i_start,
    input  wire signed [NB_W-1:0]    i_X0_re,
    input  wire signed [NB_W-1:0]    i_X0_im,
    input  wire signed [NB_W-1:0]    i_X1_re,
    input  wire signed [NB_W-1:0]    i_X1_im,

    // --- Puerto de escritura de pesos (desde LMS) ---
    input  wire                      i_we,
    input  wire [$clog2(NFFT)-1:0]   i_wk,
    input  wire                      i_wsel,    // 0=W0, 1=W1
    input  wire signed [NB_W-1:0]    i_W_re,
    input  wire signed [NB_W-1:0]    i_W_im,

    // --- Salida: hacia la IFFT ---
    output reg                       o_valid,
    output reg                       o_start,
    output reg  signed [NB_W-1:0]    o_Y_re,
    output reg  signed [NB_W-1:0]    o_Y_im
);

    // ============================================================
    // RAM de pesos W0 y W1
    // Inicializadas: W0[k]=1.0 en Q(NBF_W), W1[k]=0
    // ============================================================
    reg signed [NB_W-1:0] W0_re [0:NFFT-1];
    reg signed [NB_W-1:0] W0_im [0:NFFT-1];
    reg signed [NB_W-1:0] W1_re [0:NFFT-1];
    reg signed [NB_W-1:0] W1_im [0:NFFT-1];

    integer init_i;
    initial begin
        for (init_i = 0; init_i < NFFT; init_i = init_i + 1) begin
            W0_re[init_i] = (1 << NBF_W);  // 1.0 en Q(NBF_W)
            W0_im[init_i] = 0;
            W1_re[init_i] = 0;
            W1_im[init_i] = 0;
        end
    end

    // ============================================================
    // Contador de bin k
    // En i_start=1 el bin actual es 0, el próximo es 1
    // ============================================================
    localparam integer KW = $clog2(NFFT);
    reg [KW-1:0] k_cnt;

    always @(posedge clk) begin
        if (rst) begin
            k_cnt <= {KW{1'b0}};
        end else if (i_valid) begin
            if (i_start)
                k_cnt <= {{(KW-1){1'b0}}, 1'b1};
            else
                k_cnt <= (k_cnt == NFFT-1) ? {KW{1'b0}} : k_cnt + 1'b1;
        end
    end

    // ============================================================
    // Puerto de escritura de pesos
    // ============================================================
    always @(posedge clk) begin
        if (i_we) begin
            if (i_wsel == 1'b0) begin
                W0_re[i_wk] <= i_W_re;
                W0_im[i_wk] <= i_W_im;
            end else begin
                W1_re[i_wk] <= i_W_re;
                W1_im[i_wk] <= i_W_im;
            end
        end
    end

    // ============================================================
    // Lectura de pesos: bin 0 en i_start, k_cnt para el resto
    // ============================================================
    wire signed [NB_W-1:0] w0_re_rd = i_start ? W0_re[0] : W0_re[k_cnt];
    wire signed [NB_W-1:0] w0_im_rd = i_start ? W0_im[0] : W0_im[k_cnt];
    wire signed [NB_W-1:0] w1_re_rd = i_start ? W1_re[0] : W1_re[k_cnt];
    wire signed [NB_W-1:0] w1_im_rd = i_start ? W1_im[0] : W1_im[k_cnt];

    // ============================================================
    // M0 = W0 · X0  (combinacional)
    // ============================================================
    wire signed [NB_W-1:0] M0_re, M0_im;

    complex_mult #(
        .NB_W (NB_W),
        .NBF_W(NBF_W)
    ) u_cm0 (
        .i_aI(w0_re_rd), .i_aQ(w0_im_rd),
        .i_bI(i_X0_re),  .i_bQ(i_X0_im),
        .o_yI(M0_re),    .o_yQ(M0_im)
    );

    // ============================================================
    // M1 = W1 · X1  (combinacional)
    // ============================================================
    wire signed [NB_W-1:0] M1_re, M1_im;

    complex_mult #(
        .NB_W (NB_W),
        .NBF_W(NBF_W)
    ) u_cm1 (
        .i_aI(w1_re_rd), .i_aQ(w1_im_rd),
        .i_bI(i_X1_re),  .i_bQ(i_X1_im),
        .o_yI(M1_re),    .o_yQ(M1_im)
    );

    // ============================================================
    // Suma Y = M0 + M1 en NB_W+1 bits
    // ============================================================
    wire signed [NB_W:0] sum_re = $signed(M0_re) + $signed(M1_re);
    wire signed [NB_W:0] sum_im = $signed(M0_im) + $signed(M1_im);

    // ============================================================
    // Saturación con sat_trunc
    // ============================================================
    wire signed [NB_W-1:0] Y_re_sat;
    wire signed [NB_W-1:0] Y_im_sat;

    sat_trunc #(
        .NB_XI(NB_W+1), .NBF_XI(NBF_W),
        .NB_XO(NB_W),   .NBF_XO(NBF_W),
        .ROUND_EVEN(0)
    ) u_sat_re (
        .i_data(sum_re),
        .o_data(Y_re_sat)
    );

    sat_trunc #(
        .NB_XI(NB_W+1), .NBF_XI(NBF_W),
        .NB_XO(NB_W),   .NBF_XO(NBF_W),
        .ROUND_EVEN(0)
    ) u_sat_im (
        .i_data(sum_im),
        .o_data(Y_im_sat)
    );

    // ============================================================
    // Registro de salida (latencia 1 ciclo)
    // ============================================================
    always @(posedge clk) begin
        if (rst) begin
            o_valid <= 1'b0;
            o_start <= 1'b0;
            o_Y_re  <= {NB_W{1'b0}};
            o_Y_im  <= {NB_W{1'b0}};
        end else begin
            o_valid <= i_valid;
            o_start <= i_start;
            o_Y_re  <= i_valid ? Y_re_sat : {NB_W{1'b0}};
            o_Y_im  <= i_valid ? Y_im_sat : {NB_W{1'b0}};
        end
    end

endmodule

`default_nettype wire