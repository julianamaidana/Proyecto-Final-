`timescale 1ns/1ps

module tb_validation_tx_ch_os_fft_ifft_buffer_hist_cmul;

  localparam integer DWIDTH    = 9;
  localparam integer SNR_WIDTH = 11;
  localparam integer NFFT      = 32;

  // 1.0 en Q7 (9b signed)
  localparam signed [DWIDTH-1:0] ONE_Q7 = 9'sd128;

  reg clk;
  reg rst;

  reg signed [SNR_WIDTH-1:0] sigma_scale;

  reg bypass_tx;
  reg signed [DWIDTH-1:0] test_data_I, test_data_Q;

  reg signed [DWIDTH-1:0] i_W0_re, i_W0_im;
  reg signed [DWIDTH-1:0] i_W1_re, i_W1_im;

  wire                    fft_valid_out;
  wire signed [DWIDTH-1:0] fft_out_I, fft_out_Q;

  wire                    hb_valid_out;
  wire [4:0]              hb_k_idx;
  wire signed [DWIDTH-1:0] hb_curr_I, hb_curr_Q;
  wire signed [DWIDTH-1:0] hb_old_I,  hb_old_Q;

  wire                    y_valid;
  wire [4:0]              y_k_idx;
  wire signed [DWIDTH-1:0] y_I, y_Q;

  top_validation dut (
    .clk(clk),
    .rst(rst),

    .sigma_scale(sigma_scale),
    .bypass_tx(bypass_tx),
    .test_data_I(test_data_I),
    .test_data_Q(test_data_Q),

    .i_W0_re(i_W0_re), .i_W0_im(i_W0_im),
    .i_W1_re(i_W1_re), .i_W1_im(i_W1_im),

    .fft_valid_out(fft_valid_out),
    .fft_out_I(fft_out_I),
    .fft_out_Q(fft_out_Q),

    .hb_valid_out(hb_valid_out),
    .hb_k_idx(hb_k_idx),
    .hb_curr_I(hb_curr_I),
    .hb_curr_Q(hb_curr_Q),
    .hb_old_I(hb_old_I),
    .hb_old_Q(hb_old_Q),

    .y_valid(y_valid),
    .y_k_idx(y_k_idx),
    .y_I(y_I),
    .y_Q(y_Q)
  );

  // clock 100MHz
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  function signed [8:0] sat9;
    input signed [31:0] x;
    begin
      if (x > 255)       sat9 = 9'sd255;
      else if (x < -256) sat9 = -9'sd256;
      else               sat9 = x[8:0];
    end
  endfunction

  function is_unknown;
    input [31:0] v;
    begin
      is_unknown = (^v === 1'bx);
    end
  endfunction

  integer frames;
  integer err;
  reg signed [DWIDTH-1:0] expI, expQ;

  always @(posedge clk) begin
    if (rst) begin
      frames <= 0;
      err    <= 0;
    end else begin
      if (y_valid) begin

        // sanity: índices alineados
        if (y_k_idx !== hb_k_idx) begin
          $display("[ERR] idx mismatch y_k=%0d hb_k=%0d @t=%0t", y_k_idx, hb_k_idx, $time);
          err <= err + 1;
        end

        // sanity: nada X/Z
        if (is_unknown({y_I,y_Q,hb_curr_I,hb_old_I,hb_k_idx,y_k_idx})) begin
          $display("[ERR] X/Z detectado @t=%0t", $time);
          err <= err + 1;
        end

        // chequeo matemático (siempre, porque W0=W1=1 => y = sat(curr+old))
        expI = sat9($signed(hb_curr_I) + $signed(hb_old_I));
        expQ = sat9($signed(hb_curr_Q) + $signed(hb_old_Q));

        if (y_I !== expI || y_Q !== expQ) begin
          $display("[ERR] k=%0d y=(%0d,%0d) exp=(%0d,%0d) curr=(%0d,%0d) old=(%0d,%0d) @t=%0t",
                    y_k_idx, y_I, y_Q, expI, expQ,
                    hb_curr_I, hb_curr_Q, hb_old_I, hb_old_Q, $time);
          err <= err + 1;
        end

        // frame completo
        if (y_k_idx == (NFFT-1)) begin
          frames <= frames + 1;
          $display("[INFO] frame=%0d done (err=%0d) @t=%0t", frames+1, err, $time);
        end
      end
    end
  end

  initial begin
    // init
    rst         = 1'b1;
    sigma_scale = 0;

    // pesos: W0=W1=1 + j0
    i_W0_re = ONE_Q7;  i_W0_im = 9'sd0;
    i_W1_re = ONE_Q7;  i_W1_im = 9'sd0;

    // TEST1: DC
    bypass_tx   = 1'b1;
    test_data_I = 9'sd100;
    test_data_Q = 9'sd0;

    repeat(20) @(posedge clk);
    rst = 1'b0;

    $display("=== INICIO SIM ===");
    $display("Reset liberado.");
    $display("--- TEST 1: DC bypass + W0=W1=1.0 ---");

    wait(frames >= 5);

    $display("--- TEST 2: SISTEMA COMPLETO (TX QPSK) ---");
    bypass_tx = 1'b0;

    wait(frames >= 15);

    $display("=== FIN SIM === frames=%0d err=%0d", frames, err);
    $stop;
  end

endmodule
