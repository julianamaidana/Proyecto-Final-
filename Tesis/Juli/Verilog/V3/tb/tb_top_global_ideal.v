`timescale 1ns/1ps

module tb_top_global_ideal;

  // ======================
  // PARAMETROS
  // ======================
  localparam integer DWIDTH    = 9;
  localparam integer SNR_WIDTH = 11;
  localparam integer N_PART    = 16;   // N
  localparam integer NFFT      = 32;   // 2N
  localparam integer IDEAL_CH  = 1;

  localparam integer NFRAMES     = 100;
  localparam integer TIMEOUT_CYC = 400000;
  localparam integer TOL_LSB     = 1;

  // ======================
  // CLK / RST
  // ======================
  reg clk, rst;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // ======================
  // DUT I/O
  // ======================
  reg  signed [SNR_WIDTH-1:0] sigma_scale;
  reg                         bypass_tx;
  reg  signed [DWIDTH-1:0]    test_data_I, test_data_Q;

  wire signed [DWIDTH-1:0]    tx_sym_I, tx_sym_Q;
  wire signed [DWIDTH-1:0]    ch_out_I, ch_out_Q;

  wire                        fft_valid_out;
  wire signed [8:0]           fft_out_I, fft_out_Q;

  wire                        ifft_valid_out;
  wire signed [8:0]           ifft_out_I, ifft_out_Q;

  wire                        hb_valid_out;
  wire [4:0]                  hb_k_idx;
  wire signed [8:0]           hb_curr_I, hb_curr_Q, hb_old_I, hb_old_Q;

  // DUT
  top_global #(
    .DWIDTH(DWIDTH),
    .SNR_WIDTH(SNR_WIDTH),
    .N_PART(N_PART),
    .NFFT(NFFT),
    .IDEAL_CH(IDEAL_CH)
  ) dut (
    .clk(clk),
    .rst(rst),
    .sigma_scale(sigma_scale),
    .bypass_tx(bypass_tx),
    .test_data_I(test_data_I),
    .test_data_Q(test_data_Q),

    .tx_sym_I(tx_sym_I),
    .tx_sym_Q(tx_sym_Q),
    .ch_out_I(ch_out_I),
    .ch_out_Q(ch_out_Q),

    .fft_valid_out(fft_valid_out),
    .fft_out_I(fft_out_I),
    .fft_out_Q(fft_out_Q),

    .ifft_valid_out(ifft_valid_out),
    .ifft_out_I(ifft_out_I),
    .ifft_out_Q(ifft_out_Q),

    .hb_valid_out(hb_valid_out),
    .hb_k_idx(hb_k_idx),
    .hb_curr_I(hb_curr_I),
    .hb_curr_Q(hb_curr_Q),
    .hb_old_I(hb_old_I),
    .hb_old_Q(hb_old_Q)
  );

  // ======================
  // HELPERS
  // ======================
  function integer iabs;
    input integer x;
    begin
      if (x < 0) iabs = -x;
      else       iabs = x;
    end
  endfunction

  task wait_cycles;
    input integer n;
    integer t;
    begin
      for (t=0; t<n; t=t+1) @(posedge clk);
    end
  endtask

  // ======================
  // SCOREBOARD
  // ======================
  reg signed [DWIDTH-1:0] prevI [0:N_PART-1];
  reg signed [DWIDTH-1:0] prevQ [0:N_PART-1];
  reg signed [DWIDTH-1:0] currI [0:N_PART-1];
  reg signed [DWIDTH-1:0] currQ [0:N_PART-1];
  reg signed [DWIDTH-1:0] expI  [0:NFFT-1];
  reg signed [DWIDTH-1:0] expQ  [0:NFFT-1];

  integer k;
  integer err_cnt;

  // ============================================================
  // NUEVA CONDICION DE AVANCE: 
  // Sincronizada con la entrada de la FIFO (50MHz reales)
  // ============================================================
  wire adv = dut.phase && !dut.fifo_full;

  // ======================
  // INIT
  // ======================
  integer f;

  initial begin
   $dumpfile("tb_top_global_ideal.vcd");
    $dumpvars(0, tb_top_global_ideal);
    
    // 1. Valores iniciales
    err_cnt     = 0;
    rst         = 1'b0; // Empezamos en 0
    sigma_scale = 'sd0;
    bypass_tx   = 1'b1; // Empezamos en bypass para no inundar la FIFO
    test_data_I = 'sd0;
    test_data_Q = 'sd0;

    for (k=0; k<N_PART; k=k+1) begin
      prevI[k] = 0;
      prevQ[k] = 0;
    end

    // 2. SECUENCIA DE RESET SINCRÓNICO (Crítico para la FIFO)
    #20;                    // Esperamos a que el clk empiece a oscilar
    @(posedge clk);
    rst = 1'b1;             // Activamos reset
    repeat(15) @(posedge clk); // Lo mantenemos por 15 ciclos (Xilinx recomienda >3)
    rst = 1'b0;             // Liberamos reset
    
    // 3. ESPERA DE ESTABILIZACIÓN
    // La FIFO tarda unos ciclos en salir de reset (wr_rst_busy/rd_rst_busy)
    repeat(30) @(posedge clk); 
    
    bypass_tx = 1'b0;       // Ahora sí, liberamos los datos del PRBS

    // 4. CORRER FRAMES
    for (f=0; f<NFRAMES; f=f+1) begin
      drive_and_capture_block(f);
      build_expected();
      capture_and_vm_frame(f);
      update_prev();

    end

    if (err_cnt == 0)
      $display("TOP_GLOBAL PRBS TEST OK: %0d frames sin errores", NFRAMES);
    else
      $display("TOP_GLOBAL PRBS TEST FAIL: errores=%0d en %0d frames", err_cnt, NFRAMES);

    $finish;
  end

  // ============================================================
  // Captura el bloque "curr" basado en lo que ENTRA a la FIFO
  // ============================================================
  task drive_and_capture_block;
    input integer frame_id;
    integer idx;
    integer t;
    begin
      idx = 0;
      t   = 0;

      while (idx < N_PART) begin
        @(posedge clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout capturando curr block frame %0d", frame_id);
          $finish;
        end

        // Si el transmisor escribió en la FIFO, capturamos el dato en el Scoreboard
        if (adv) begin
          currI[idx] = dut.to_fifo_I; // Usamos los cables que van a la FIFO
          currQ[idx] = dut.to_fifo_Q;
          idx = idx + 1;
        end
      end
    end
  endtask

  // ======================
  // exp = {prev, curr}
  // ======================
  task build_expected;
    begin
      for (k=0; k<N_PART; k=k+1) begin
        expI[k]        = prevI[k];
        expQ[k]        = prevQ[k];
        expI[k+N_PART] = currI[k];
        expQ[k+N_PART] = currQ[k];
      end
    end
  endtask

  task update_prev;
    begin
      for (k=0; k<N_PART; k=k+1) begin
        prevI[k] = currI[k];
        prevQ[k] = currQ[k];
      end
    end
  endtask

  // ======================
  // Espera ifft_start y VM de 32 samples
  // ======================
  task capture_and_vm_frame;
    input integer frame_id;
    integer idx;
    integer t;
    integer di, dq;
    reg seen_start;
    begin
      t = 0;
      seen_start = 0;

      while (!seen_start) begin
        @(posedge clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout esperando ifft_start frame %0d", frame_id);
          $finish;
        end

        // Detectar arranque del bloque de salida
        if ((dut.ifft_start_w === 1'b1) || (ifft_valid_out === 1'b1))
          seen_start = 1'b1;
      end

      idx = 0;
      t   = 0;

      while (idx < NFFT) begin
        @(posedge clk);
        t = t + 1;
        if (t > TIMEOUT_CYC) begin
          $display("ERROR: timeout capturando ifft frame %0d idx=%0d", frame_id, idx);
          $finish;
        end

        if (ifft_valid_out) begin
          di = $signed(ifft_out_I) - $signed(expI[idx]);
          dq = $signed(ifft_out_Q) - $signed(expQ[idx]);

          if ((iabs(di) > TOL_LSB) || (iabs(dq) > TOL_LSB)) begin
            err_cnt = err_cnt + 1;
            $display("ERROR: mismatch frame %0d idx %0d @t=%0t", frame_id, idx, $time);
            $display("   got: I=%0d Q=%0d", $signed(ifft_out_I), $signed(ifft_out_Q));
            $display("   exp: I=%0d Q=%0d", $signed(expI[idx]), $signed(expQ[idx]));
          end
          idx = idx + 1;
        end
      end
      $display("OK: frame %0d VM OK @t=%0t", frame_id, $time);
    end
  endtask

endmodule
