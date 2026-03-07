`timescale 1ns/1ps

module top_validation #(
    parameter integer DWIDTH    = 9,
    parameter integer SNR_WIDTH = 11,
    parameter integer N_PART    = 16,
    parameter integer NFFT      = 32
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire signed [SNR_WIDTH-1:0] sigma_scale,
    input  wire                        bypass_tx,
    input  wire signed [DWIDTH-1:0]    test_data_I,
    input  wire signed [DWIDTH-1:0]    test_data_Q,
    input  wire                        test_data_valid, 

    // Coeficientes
    input  wire signed [DWIDTH-1:0]    i_W0_re, i_W0_im,
    input  wire signed [DWIDTH-1:0]    i_W1_re, i_W1_im,

    // Salidas de Monitoreo
    output wire                        fft_valid_out,
    output wire signed [DWIDTH-1:0]    fft_out_I, fft_out_Q,
    output wire                        y_valid,
    output wire signed [DWIDTH-1:0]    y_I, y_Q,
    output wire                        ifft_valid_out,
    output wire signed [DWIDTH-1:0]    ifft_out_I, ifft_out_Q,
    
    // NUEVA SALIDA DE DEBUG PARA FIFO
    output wire                        o_fifo_full_dbg
);

    // 1) TX+CANAL
    wire signed [DWIDTH-1:0] tx_rx_I, tx_rx_Q;
    wire fifo_full;

    top #(.DWIDTH(DWIDTH), .SNR_WIDTH(SNR_WIDTH)) u_system_original (
        .clk(clk), .rst(rst), .sigma_scale(sigma_scale), .i_backpressure(fifo_full),
        .rx_I(tx_rx_I), .rx_Q(tx_rx_Q)
    );

    // 2) SELECCIÓN DE FUENTE (Bypass o Real)
    wire signed [DWIDTH-1:0] src_I = (bypass_tx) ? test_data_I : tx_rx_I;
    wire signed [DWIDTH-1:0] src_Q = (bypass_tx) ? test_data_Q : tx_rx_Q;
    // En modo real (bypass=0), asumimos valid continuo (1'b1)
    wire                     src_valid = (bypass_tx) ? test_data_valid : 1'b1;

    // -----------------------------------------------------------
    // INTEGRACIÓN DE LA FIFO
    // -----------------------------------------------------------
    wire fifo_full, fifo_empty;
    wire [2*DWIDTH-1:0] fifo_din  = {src_I, src_Q};
    wire [2*DWIDTH-1:0] fifo_dout;
    wire fifo_wr_en = src_valid && !fifo_full; // Escribir si hay dato y hay lugar
    wire fifo_rd_en; // Se define más abajo con el handshake del buffer

    // Sacamos la señal full para verla en el testbench
    assign o_fifo_full_dbg = fifo_full;

    simple_fifo #(.DATA_WIDTH(2*DWIDTH), .DEPTH(1024)) u_input_fifo (
        .clk(clk), .rst(rst),
        .i_wr_en(fifo_wr_en), .i_data(fifo_din), .o_full(fifo_full),
        .i_rd_en(fifo_rd_en), .o_data(fifo_dout), .o_empty(fifo_empty),
        .count() // No conectado
    );

    // Desempaquetar salida de FIFO
    wire signed [DWIDTH-1:0] fifo_out_I = fifo_dout[2*DWIDTH-1 : DWIDTH];
    wire signed [DWIDTH-1:0] fifo_out_Q = fifo_dout[DWIDTH-1 : 0];

    // Lógica de Lectura (Handshake):
    // Leemos de FIFO si NO está vacía Y el os_buffer está listo (o_in_ready)
    wire os_ready_w;
    assign fifo_rd_en = (!fifo_empty) && os_ready_w;

    // -----------------------------------------------------------
    // 3) OVERLAP-SAVE BUFFER (Conectado a la FIFO)
    // -----------------------------------------------------------
    wire signed [DWIDTH-1:0] os_out_I, os_out_Q;
    wire                     os_valid_w;

    os_buffer #(.N(N_PART), .WN(DWIDTH)) u_buffer (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (fifo_rd_en),   // Valid solo si la FIFO entregó dato
        .i_xI        (fifo_out_I),   // Datos desde FIFO
        .i_xQ        (fifo_out_Q),
        .o_in_ready  (os_ready_w),   // Controla la lectura de la FIFO
        .o_fft_start (),
        .o_fft_valid (os_valid_w),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q)
    );

    // 4) FFT
    fft_ifft #(.NFFT(NFFT), .NB_IN(DWIDTH), .NB_OUT(DWIDTH), .SCALE_STAGE(0)) u_fft (
        .i_clk(clk), .i_rst(rst),
        .i_valid(os_valid_w), .i_xI(os_out_I), .i_xQ(os_out_Q), .i_inverse(1'b0),
        .o_in_ready(), .o_start(), .o_valid(fft_valid_out), .o_yI(fft_out_I), .o_yQ(fft_out_Q)
    );

    // 5) HISTORY BUFFER
    wire hb_valid_out;
    wire [4:0] hb_k_idx;
    wire signed [DWIDTH-1:0] hb_curr_I, hb_curr_Q, hb_old_I, hb_old_Q;
    history_buffer #(.W(DWIDTH)) u_history (
        .clk(clk), .rst(rst),
        .i_valid(fft_valid_out), .i_X_re(fft_out_I), .i_X_im(fft_out_Q),
        .i_W0_re(i_W0_re), .i_W0_im(i_W0_im), .i_W1_re(i_W1_re), .i_W1_im(i_W1_im),
        .o_valid_data(hb_valid_out), .o_k_idx(hb_k_idx),
        .o_X_curr_re(hb_curr_I), .o_X_curr_im(hb_curr_Q), .o_X_old_re(hb_old_I), .o_X_old_im(hb_old_Q)
    );

    // 6) CMUL + SUMA
    wire signed [DWIDTH-1:0] y0_I, y0_Q, y1_I, y1_Q;
    complex_mult #(.NB_W(DWIDTH), .NBF_W(7)) u_cmul0 (.i_aI(hb_curr_I), .i_aQ(hb_curr_Q), .i_bI(i_W0_re), .i_bQ(i_W0_im), .o_yI(y0_I), .o_yQ(y0_Q));
    complex_mult #(.NB_W(DWIDTH), .NBF_W(7)) u_cmul1 (.i_aI(hb_old_I), .i_aQ(hb_old_Q), .i_bI(i_W1_re), .i_bQ(i_W1_im), .o_yI(y1_I), .o_yQ(y1_Q));
    
    wire signed [31:0] sumI = $signed(y0_I) + $signed(y1_I);
    wire signed [31:0] sumQ = $signed(y0_Q) + $signed(y1_Q);
    assign y_I = (sumI > 255) ? 9'sd255 : (sumI < -256) ? -9'sd256 : sumI[8:0];
    assign y_Q = (sumQ > 255) ? 9'sd255 : (sumQ < -256) ? -9'sd256 : sumQ[8:0];
    assign y_valid = hb_valid_out;

    // 7) IFFT
    fft_ifft #(.NFFT(NFFT), .NB_IN(DWIDTH), .NB_OUT(DWIDTH), .SCALE_STAGE(0)) u_ifft (
        .i_clk(clk), .i_rst(rst),
        .i_valid(y_valid), .i_xI(y_I), .i_xQ(y_Q), .i_inverse(1'b1),
        .o_in_ready(), .o_start(), .o_valid(ifft_valid_out), .o_yI(ifft_out_I), .o_yQ(ifft_out_Q)
    );

endmodule