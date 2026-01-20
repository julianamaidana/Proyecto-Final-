`timescale 1ns/1ps

module tb_tx_ch_os_fft_ifft;

    parameter integer DWIDTH    = 9;
    parameter integer DATA_F    = 7;
    parameter integer SNR_WIDTH = 11;
    parameter integer OS_N      = 16;

    localparam integer NFFT    = 32;
    localparam integer TOL_LSB = 4;

    // FIFO depth de frames esperados
    localparam integer DEPTH = 8;

    reg clk;
    reg rst;
    reg signed [SNR_WIDTH-1:0] sigma_scale;

    wire signed [DWIDTH-1:0] rx_I, rx_Q;

    wire os_in_ready;
    wire os_fft_start;
    wire os_fft_valid;
    wire signed [DWIDTH-1:0] os_fft_xI, os_fft_xQ;

    wire fft_in_ready;
    wire fft_start, fft_valid;
    wire signed [16:0] fft_yI_w, fft_yQ_w;

    wire ifft_in_ready;
    wire ifft_start, ifft_valid;
    wire signed [DWIDTH-1:0] ifft_yI, ifft_yQ;

    // DUT
    top_tx_ch_os_fft_ifft #(
        .DWIDTH(DWIDTH),
        .DATA_F(DATA_F),
        .SNR_WIDTH(SNR_WIDTH),
        .OS_N(OS_N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .sigma_scale(sigma_scale),

        .rx_I(rx_I),
        .rx_Q(rx_Q),

        .os_in_ready(os_in_ready),
        .os_fft_start(os_fft_start),
        .os_fft_valid(os_fft_valid),
        .os_fft_xI(os_fft_xI),
        .os_fft_xQ(os_fft_xQ),

        .fft_in_ready(fft_in_ready),
        .fft_start(fft_start),
        .fft_valid(fft_valid),
        .fft_yI_w(fft_yI_w),
        .fft_yQ_w(fft_yQ_w),

        .ifft_in_ready(ifft_in_ready),
        .ifft_start(ifft_start),
        .ifft_valid(ifft_valid),
        .ifft_yI(ifft_yI),
        .ifft_yQ(ifft_yQ)
    );

    // Clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ============================================================
    // Helpers
    // ============================================================
    function integer iabs;
        input integer x;
        begin
            if (x < 0) iabs = -x;
            else       iabs = x;
        end
    endfunction

    // ============================================================
    // FIFO de frames esperados (Verilog puro -> linealizado)
    // exp_fifo_I[frame*NFFT + idx]
    // ============================================================
    reg signed [DWIDTH-1:0] exp_fifo_I [0:(DEPTH*NFFT)-1];
    reg signed [DWIDTH-1:0] exp_fifo_Q [0:(DEPTH*NFFT)-1];

    integer wr_ptr;        // 0..DEPTH-1
    integer rd_ptr;        // 0..DEPTH-1
    integer fifo_count;    // 0..DEPTH

    integer wr_idx;        // 0..NFFT-1 (dentro del frame al capturar)
    integer rd_idx;        // 0..NFFT-1 (dentro del frame al comparar)
    integer active_base;   // base = popped_ptr*NFFT

    integer err_cnt;
    integer frames_ok;
    integer dI, dQ;

    integer k;

    // ============================================================
    // Captura OS -> FIFO
    // ============================================================
    always @(posedge clk) begin
        if (rst) begin
            wr_ptr     <= 0;
            rd_ptr     <= 0;
            fifo_count <= 0;
            wr_idx     <= 0;
        end else begin
            // inicio de frame OS
            if (os_fft_start) begin
                wr_idx <= 0;
            end

            if (os_fft_valid) begin
                // si FIFO lleno -> error (no debería pasar si el sistema va “de a frame”)
                if (fifo_count >= DEPTH) begin
                    $display("[ERROR] FIFO de esperados LLENO! Se pierde frame OS @t=%0t", $time);
                    err_cnt <= err_cnt + 1;
                end else begin
                    exp_fifo_I[wr_ptr*NFFT + wr_idx] <= os_fft_xI;
                    exp_fifo_Q[wr_ptr*NFFT + wr_idx] <= os_fft_xQ;

                    if (wr_idx == (NFFT-1)) begin
                        // cerramos frame
                        wr_idx <= 0;
                        wr_ptr <= (wr_ptr == (DEPTH-1)) ? 0 : (wr_ptr + 1);
                        fifo_count <= fifo_count + 1;
                    end else begin
                        wr_idx <= wr_idx + 1;
                    end
                end
            end
        end
    end

    // ============================================================
    // Pop en ifft_start: selecciona frame esperado correcto
    // ============================================================
    always @(posedge clk) begin
        if (rst) begin
            rd_idx      <= 0;
            active_base <= 0;
        end else begin
            if (ifft_start) begin
                if (fifo_count == 0) begin
                    $display("[ERROR] IFFT_START pero FIFO VACIO! @t=%0t", $time);
                    err_cnt <= err_cnt + 1;
                    // igual arranco base=0, pero va a fallar
                    active_base <= 0;
                    rd_idx <= 0;
                end else begin
                    // “pop” frame
                    active_base <= rd_ptr * NFFT;
                    rd_ptr <= (rd_ptr == (DEPTH-1)) ? 0 : (rd_ptr + 1);
                    fifo_count <= fifo_count - 1;
                    rd_idx <= 0;
                end
            end

            if (ifft_valid) begin
                // comparo contra el frame activo
                dI = $signed(ifft_yI) - $signed(exp_fifo_I[active_base + rd_idx]);
                dQ = $signed(ifft_yQ) - $signed(exp_fifo_Q[active_base + rd_idx]);

                if ((iabs(dI) > TOL_LSB) || (iabs(dQ) > TOL_LSB)) begin
                    $display("[ERROR] IFFT mismatch idx=%0d @t=%0t got(I=%0d,Q=%0d) exp(I=%0d,Q=%0d) dI=%0d dQ=%0d",
                             rd_idx, $time,
                             $signed(ifft_yI), $signed(ifft_yQ),
                             $signed(exp_fifo_I[active_base + rd_idx]),
                             $signed(exp_fifo_Q[active_base + rd_idx]),
                             dI, dQ);
                    err_cnt <= err_cnt + 1;
                end

                if (rd_idx == (NFFT-1)) begin
                    rd_idx <= 0;
                    frames_ok <= frames_ok + 1;
                    $display("[INFO] Termino IFFT frame OKcount=%0d err=%0d @t=%0t", frames_ok, err_cnt, $time);
                end else begin
                    rd_idx <= rd_idx + 1;
                end
            end
        end
    end

    // ============================================================
    // Init + stimulus
    // ============================================================
    initial begin
        err_cnt   = 0;
        frames_ok = 0;

        // init fifo memories a 0 (no estrictamente necesario)
        for (k = 0; k < DEPTH*NFFT; k = k + 1) begin
            exp_fifo_I[k] = {DWIDTH{1'b0}};
            exp_fifo_Q[k] = {DWIDTH{1'b0}};
        end

        rst = 1;
        sigma_scale = 0;

        #100;
        rst = 0;

        #5000;
        sigma_scale = 11'd200;
        #5000;
        sigma_scale = 11'd800;
        #5000;

        $display("=====================================");
        $display("FIN SIM: frames_ok=%0d  err_cnt=%0d", frames_ok, err_cnt);
        $display("=====================================");
        $stop;
    end

endmodule
