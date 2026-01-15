`timescale 1ns/1ps

module tb_top;

    parameter DWIDTH    = 9;
    parameter SNR_WIDTH = 11;
    parameter DATA_F    = 7;   // para ver en real (Q7)
    parameter OS_N      = 16;

    reg clk;
    reg rst;
    reg signed [SNR_WIDTH-1:0] sigma_scale;

    wire signed [DWIDTH-1:0] rx_I;
    wire signed [DWIDTH-1:0] rx_Q;

    wire os_in_ready;
    wire os_fft_start;
    wire os_fft_valid;
    wire signed [DWIDTH-1:0] os_fft_xI;
    wire signed [DWIDTH-1:0] os_fft_xQ;

    // DUT
    top_tx_ch_os #(
        .DWIDTH(DWIDTH),
        .SNR_WIDTH(SNR_WIDTH),
        .OS_N(OS_N)
    ) dut (
        .clk         (clk),
        .rst         (rst),
        .sigma_scale (sigma_scale),

        .rx_I        (rx_I),
        .rx_Q        (rx_Q),

        .os_in_ready (os_in_ready),
        .os_fft_start(os_fft_start),
        .os_fft_valid(os_fft_valid),
        .os_fft_xI   (os_fft_xI),
        .os_fft_xQ   (os_fft_xQ)
    );

    // ============================================================
    // Clock
    // ============================================================
    initial clk = 0;
    always #5 clk = ~clk;

    // ============================================================
    // Monitoreo por bloques 2N
    // ============================================================
    integer k;
    integer idx;

    reg signed [DWIDTH-1:0] blk_I [0:(2*OS_N)-1];
    reg signed [DWIDTH-1:0] blk_Q [0:(2*OS_N)-1];

    reg signed [DWIDTH-1:0] prev_new_I [0:OS_N-1];
    reg signed [DWIDTH-1:0] prev_new_Q [0:OS_N-1];

    integer block_count;
    integer err_cnt;

    initial begin
        for (k = 0; k < OS_N; k = k + 1) begin
            prev_new_I[k] = {DWIDTH{1'b0}};
            prev_new_Q[k] = {DWIDTH{1'b0}};
        end
        block_count = 0;
        err_cnt = 0;
    end

    // Captura de stream OS
    always @(posedge clk) begin
        if (rst) begin
            idx <= 0;
        end else begin
            if (os_fft_start) begin
                idx <= 0;
            end

            if (os_fft_valid) begin
                blk_I[idx] <= os_fft_xI;
                blk_Q[idx] <= os_fft_xQ;
                idx <= idx + 1;
            end

            // Cuando termina un bloque: idx llegó a 2N
            if (!rst && os_fft_valid && (idx == (2*OS_N-1))) begin
                // chequeos
                if (block_count == 0) begin
                    // Primer bloque: overlap debería ser cero
                    for (k = 0; k < OS_N; k = k + 1) begin
                        if (blk_I[k] !== {DWIDTH{1'b0}} || blk_Q[k] !== {DWIDTH{1'b0}}) begin
                            $display("[ERROR] Bloque0 overlap no-cero en k=%0d: I=%0d Q=%0d",
                                     k, blk_I[k], blk_Q[k]);
                            err_cnt = err_cnt + 1;
                        end
                    end
                end else begin
                    // Bloques siguientes: overlap == prev_new
                    for (k = 0; k < OS_N; k = k + 1) begin
                        if (blk_I[k] !== prev_new_I[k] || blk_Q[k] !== prev_new_Q[k]) begin
                            $display("[ERROR] Bloque%0d overlap mismatch k=%0d: got(I=%0d,Q=%0d) exp(I=%0d,Q=%0d)",
                                     block_count, k, blk_I[k], blk_Q[k], prev_new_I[k], prev_new_Q[k]);
                            err_cnt = err_cnt + 1;
                        end
                    end
                end

                // Guardamos "new" actual para el próximo bloque
                for (k = 0; k < OS_N; k = k + 1) begin
                    prev_new_I[k] <= blk_I[OS_N + k];
                    prev_new_Q[k] <= blk_Q[OS_N + k];
                end

                block_count <= block_count + 1;

                $display("[INFO] Termino bloque %0d (err acumulados=%0d)", block_count, err_cnt);
            end
        end
    end

    // ============================================================
    // Conversión a real (para graficar en wave)
    // ============================================================
    real SCALE_FACTOR;
    real real_rx_I, real_rx_Q;
    real real_os_I, real_os_Q;

    initial SCALE_FACTOR = 2.0 ** DATA_F;

    always @* begin
        real_rx_I = $itor(rx_I) / SCALE_FACTOR;
        real_rx_Q = $itor(rx_Q) / SCALE_FACTOR;

        real_os_I = $itor(os_fft_xI) / SCALE_FACTOR;
        real_os_Q = $itor(os_fft_xQ) / SCALE_FACTOR;
    end

    // ============================================================
    // Stimulus
    // ============================================================
    initial begin
        rst = 1;
        sigma_scale = 0;

        #100;
        rst = 0;

        // Fase limpia
        #5000;

        // Ruido bajo
        sigma_scale = 11'd200;
        #5000;

        // Ruido alto
        sigma_scale = 11'd800;
        #5000;

        $display("=====================================");
        $display("FIN SIM: blocks=%0d  err_cnt=%0d", block_count, err_cnt);
        $display("=====================================");
        $stop;
    end

endmodule
