`timescale 1ns/1ps

module top_validation #(
    parameter DWIDTH    = 9,
    parameter SNR_WIDTH = 11,
    parameter N_PART    = 16,
    parameter NFFT      = 32
)(
    input  wire        clk,
    input  wire        rst,
    input  wire signed [SNR_WIDTH-1:0] sigma_scale,
    input  wire        tb_tx_en,
    input  wire        i_valid_from_tb,    // Puerto para manejar latencia desde TB

    // Puertos de monitoreo (Los que daban error en la imagen)
    output wire        o_check_valid,
    output wire [8:0]  o_check_I,
    output wire [8:0]  o_check_Q,

    // Salidas finales
    output wire        o_clean_valid,
    output wire [8:0]  o_clean_I,
    output wire [8:0]  o_clean_Q,
    output wire        buf_ready_out
);
    // ====================================================
    // CABLES INTERNOS
    // ====================================================
    // Etapa 1: Sistema (TX + Canal)
    wire [DWIDTH-1:0] sys_I, sys_Q;
    wire              sys_valid;

    // Etapa 2: Buffer de Overlap
    wire [DWIDTH-1:0] os_out_I, os_out_Q;
    wire              os_valid;
    wire              fft_ready;

    // Etapa 3: FFT
    wire [8:0]        fft_out_I, fft_out_Q;
    wire              fft_out_valid;

    // Etapa 4: Receptor (History + Mult)
    wire              hb_valid;
    wire [8:0]        hb_curr_I, hb_curr_Q;
    wire [8:0]        hb_old_I,  hb_old_Q;
    wire [8:0]        prod0_I, prod0_Q, prod1_I, prod1_Q;
    wire [8:0]        mult_out_I, mult_out_Q;

    // Etapa 5: IFFT + Discard
    wire              ifft_v, ifft_s;
    wire [8:0]        ifft_yI, ifft_yQ;
    wire              ifft_last;
    reg  [4:0]        cnt_ifft;

    // ====================================================
    // 1. SISTEMA (TX + CANAL)
    // ====================================================
    top #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH)
    ) u_system (
        .i_clk         (clk),
        .i_rst         (rst),
        .i_sigma_scale (sigma_scale),
        .i_tx_en       (tb_tx_en & buf_ready_out),    // Controlado por el TB
        .o_rx_I        (sys_I),
        .o_rx_Q        (sys_Q),
        .o_rx_valid    (sys_valid)    //
    );

    // ====================================================
    // 2. BUFFER DE OVERLAP-SAVE
    // ====================================================
    os_buffer #(
        .N (N_PART),
        .WN(DWIDTH)
    ) u_os (
        .i_clk       (clk),
        .i_rst       (rst),
        .i_valid     (i_valid_from_tb),
        .i_xI        (sys_I),
        .i_xQ        (sys_Q),
        .i_fft_ready (fft_ready),

        .o_in_ready  (buf_ready_out), // Hacia el exterior
        .o_fft_valid (os_valid),
        .o_fft_xI    (os_out_I),
        .o_fft_xQ    (os_out_Q)
    );

    // ====================================================
    // 3. FFT PRINCIPAL
    // ====================================================
    fft_ifft #(
        .NFFT(NFFT), .NB_IN(DWIDTH), .NB_OUT(9), .NB_W(24)
    ) u_fft (
        .i_clk(clk), .i_rst(rst),
        .i_valid(os_valid),
        .i_xI(os_out_I), .i_xQ(os_out_Q),
        .o_in_ready(fft_ready), 
        .o_valid(fft_out_valid),
        .o_yI(fft_out_I), .o_yQ(fft_out_Q)
    );


    // ====================================================
    // 4. IFFT DE PRUEBA (SOLO PARA VALIDAR FFT)
    // ====================================================
    fft_ifft #(
        .NFFT(NFFT), .NB_IN(9), .NB_OUT(9), .NB_W(24), .SCALE_STAGE(0)
    ) u_ifft_check (
        .i_clk(clk), 
        .i_rst(rst),
        .i_inverse(1'b1),        // Modo Inverso
        .i_valid(fft_out_valid), 
        .i_xI(fft_out_I), 
        .i_xQ(fft_out_Q),
        .o_yI(o_check_I),
        .o_yQ(o_check_Q),
        .o_valid(o_check_valid),
        .o_in_ready(), .o_start()
    );
    // ====================================================
    // 4. HISTORY BUFFER (FDE)
    // ====================================================
    history_buffer #(.W(9)) u_hist (
        .clk(clk),
        .rst(rst),
        .i_valid(fft_out_valid),
        .i_X_re(fft_out_I),
        .i_X_im(fft_out_Q),
        .i_W0_re(9'sd0), .i_W0_im(9'sd0),
        .i_W1_re(9'sd0), .i_W1_im(9'sd0),
        .o_valid_data(hb_valid),
        .o_X_curr_re(hb_curr_I), .o_X_curr_im(hb_curr_Q),
        .o_X_old_re (hb_old_I),  .o_X_old_im (hb_old_Q),
        .o_k_idx(), .o_W0_re(), .o_W0_im(), .o_W1_re(), .o_W1_im()
    );

    // ====================================================
    // 5. ECUALIZACIÓN (PRODUCTOS COMPLEJOS)
    // ====================================================
    complex_mult #( .NB_W(9), .NBF_W(7) ) u_mult0 (
        .i_aI(hb_curr_I), .i_aQ(hb_curr_Q),
        .i_bI(9'sd128),   .i_bQ(9'sd0),     // Coeficiente 1.0 en Q1.7
        .o_yI(prod0_I),   .o_yQ(prod0_Q)
    );

    complex_mult #( .NB_W(9), .NBF_W(7) ) u_mult1 (
        .i_aI(hb_old_I), .i_aQ(hb_old_Q),
        .i_bI(9'sd0),    .i_bQ(9'sd0),      // Rama de historia anulada
        .o_yI(prod1_I),  .o_yQ(prod1_Q)
    );

    assign mult_out_I = prod0_I + prod1_I;
    assign mult_out_Q = prod0_Q + prod1_Q;

    // ====================================================
    // 6. IFFT (RETORNO AL TIEMPO)
    // ====================================================
    fft_ifft #(
        .NFFT(NFFT), .NB_IN(9), .NB_OUT(9), .NB_W(24), .SCALE_STAGE(0)
    ) u_ifft (
        .i_clk(clk), 
        .i_rst(rst),
        .i_inverse(1'b1), 
        .i_valid(hb_valid),
        .i_xI(mult_out_I), 
        .i_xQ(mult_out_Q),
        .o_in_ready(), 
        .o_start(ifft_s),
        .o_valid(ifft_v),
        .o_yI(ifft_yI), 
        .o_yQ(ifft_yQ)
    );

    // ====================================================
    // 7. GENERACIÓN DE SEÑAL LAST Y DISCARD HALF
    // ====================================================
    always @(posedge clk) begin
        if (rst)          cnt_ifft <= 0;
        else if (ifft_s)  cnt_ifft <= 0;
        else if (ifft_v)  cnt_ifft <= cnt_ifft + 1;
    end
    
    assign ifft_last = (cnt_ifft == 5'd31) && ifft_v;

    discard_half #(
        .NFFT(32), .W(9), .DISCARD(16)
    ) u_discard (
        .i_clk(clk), .i_rst(rst),
        .i_valid(ifft_v),
        .i_y_re(ifft_yI <<< 5),  // Ganancia compensatoria
        .i_y_im(ifft_yQ <<< 5),
        .i_last(ifft_last),
        
        .o_valid(o_clean_valid),
        .o_y_re (o_clean_I),
        .o_y_im (o_clean_Q),
        .o_last (), .o_first(), .o_idx()
    );

endmodule