`timescale 1ns / 1ps

module tb_validation;

    reg clk = 0;
    reg rst = 1;

    reg signed [10:0] sigma_scale = 11'sd0;

    reg bypass_tx = 0;
    reg signed [8:0] test_data_I = 0;
    reg signed [8:0] test_data_Q = 0;

    wire fft_valid_out;
    wire signed [8:0] fft_out_I;
    wire signed [8:0] fft_out_Q;

    // NUEVO: history buffer
    wire hb_valid_out;
    wire [4:0] hb_k_idx;
    wire signed [8:0] hb_curr_I, hb_curr_Q;
    wire signed [8:0] hb_old_I,  hb_old_Q;

    always #5 clk = ~clk;

    top_validation #(
        .DWIDTH(9),
        .SNR_WIDTH(11),
        .N_PART(16),
        .NFFT(32)
    ) u_dut (
        .clk(clk),
        .rst(rst),
        .sigma_scale(sigma_scale),
        .bypass_tx(bypass_tx),
        .test_data_I(test_data_I),
        .test_data_Q(test_data_Q),

        .fft_valid_out(fft_valid_out),
        .fft_out_I(fft_out_I),
        .fft_out_Q(fft_out_Q),

        .hb_valid_out(hb_valid_out),
        .hb_k_idx(hb_k_idx),
        .hb_curr_I(hb_curr_I),
        .hb_curr_Q(hb_curr_Q),
        .hb_old_I(hb_old_I),
        .hb_old_Q(hb_old_Q)
    );

    // ============================================================
    // SCOREBOARD HISTORY BUFFER
    // ============================================================
    reg signed [8:0] prev_I [0:31];
    reg signed [8:0] prev_Q [0:31];

    integer i;
    integer frame_cnt;
    integer hb_err;

    initial begin
        for (i=0; i<32; i=i+1) begin
            prev_I[i] = 0;
            prev_Q[i] = 0;
        end
        frame_cnt = 0;
        hb_err = 0;
    end

    // contamos bins dentro de cada frame
    integer bin_cnt;

    always @(posedge clk) begin
        if (rst) begin
            bin_cnt <= 0;
            frame_cnt <= 0;
            hb_err <= 0;
            for (i=0; i<32; i=i+1) begin
                prev_I[i] <= 0;
                prev_Q[i] <= 0;
            end
        end else begin
            if (hb_valid_out) begin
                // hb_k_idx está alineado (lo arreglamos en history_buffer.v)
                // Para el frame 0, old debe ser 0
                if (frame_cnt == 0) begin
                    if (hb_old_I !== 9'sd0 || hb_old_Q !== 9'sd0) begin
                        $display("[ERROR] HB frame0 k=%0d old no-cero: oldI=%0d oldQ=%0d @t=%0t",
                                 hb_k_idx, hb_old_I, hb_old_Q, $time);
                        hb_err = hb_err + 1;
                    end
                end else begin
                    // Para frame>=1: old(k) debe ser el curr(k) del frame anterior
                    if (hb_old_I !== prev_I[hb_k_idx] || hb_old_Q !== prev_Q[hb_k_idx]) begin
                        $display("[ERROR] HB mismatch frame=%0d k=%0d got_old(I=%0d,Q=%0d) exp_old(I=%0d,Q=%0d) @t=%0t",
                                 frame_cnt, hb_k_idx,
                                 hb_old_I, hb_old_Q,
                                 prev_I[hb_k_idx], prev_Q[hb_k_idx],
                                 $time);
                        hb_err = hb_err + 1;
                    end
                end

                // guardo el curr actual para usarlo como "prev" en el próximo frame
                prev_I[hb_k_idx] <= hb_curr_I;
                prev_Q[hb_k_idx] <= hb_curr_Q;

                // bin_cnt sólo para detectar fin de frame
                if (bin_cnt == 31) begin
                    bin_cnt <= 0;
                    frame_cnt <= frame_cnt + 1;
                    $display("[INFO] Termino frame FFT/HB frame=%0d (hb_err=%0d) @t=%0t",
                             frame_cnt, hb_err, $time);
                end else begin
                    bin_cnt <= bin_cnt + 1;
                end
            end
        end
    end

    // ============================================================
    // STIMULUS (tu prueba original, intacta)
    // ============================================================
    initial begin
        $display("=== INICIO SIMULACION ===");
        clk = 0;
        rst = 1;
        bypass_tx = 0;
        sigma_scale = 0;

        #100;
        rst = 0;
        $display("Reset liberado.");

        // TEST 1: constante DC
        $display("--- TEST 1: INYECCION DE CONSTANTE (DC) ---");
        bypass_tx   = 1;
        test_data_I = 9'sd100;
        test_data_Q = 9'sd0;

        repeat (5) @(posedge fft_valid_out);
        $display("Ya deberían haber salido datos de la FFT (Mirar Waveform).");

        // TEST 2: sistema completo QPSK
        $display("--- TEST 2: SISTEMA COMPLETO (TX QPSK) ---");
        #200;
        bypass_tx = 0;

        #10000;

        $display("=== FIN SIMULACION ===");
        $display("HB frames=%0d  hb_err=%0d", frame_cnt, hb_err);
        $stop;
    end

endmodule
