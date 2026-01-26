`timescale 1ns/1ps

module tb_validation_tx_ch_os_fft_ifft_buffer_hist_cmul;

  localparam integer DWIDTH    = 9;
  localparam integer SNR_WIDTH = 11;
  localparam integer NFFT      = 32;

  // Q7
  localparam signed [DWIDTH-1:0] ONE_Q7  = 9'sd128;
  localparam signed [DWIDTH-1:0] ZERO_Q7 = 9'sd0;

  reg clk;
  reg rst;

  reg  signed [SNR_WIDTH-1:0] sigma_scale;
  reg  bypass_tx;
  reg  signed [DWIDTH-1:0] test_data_I, test_data_Q;

  reg  signed [DWIDTH-1:0] i_W0_re, i_W0_im;
  reg  signed [DWIDTH-1:0] i_W1_re, i_W1_im;

  wire fft_valid_out;
  wire signed [DWIDTH-1:0] fft_out_I, fft_out_Q;

  wire hb_valid_out;
  wire [4:0] hb_k_idx;
  wire signed [DWIDTH-1:0] hb_curr_I, hb_curr_Q;
  wire signed [DWIDTH-1:0] hb_old_I,  hb_old_Q;

  wire y_valid;
  wire [4:0] y_k_idx;
  wire signed [DWIDTH-1:0] y_I, y_Q;

  // DUT
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

  // clock 100 MHz
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  // sat 9-bit signed [-256..255]
  function signed [DWIDTH-1:0] sat9;
    input signed [31:0] x;
    begin
      if (x > 255)       sat9 = 9'sd255;
      else if (x < -256) sat9 = -9'sd256;
      else               sat9 = x[DWIDTH-1:0];
    end
  endfunction

  // referencia CMUL: (a+jb)*(c+jd) con Q7 (>>7 trunc)
  function signed [DWIDTH-1:0] cmul_re_q7;
    input signed [DWIDTH-1:0] aI, aQ, bI, bQ;
    reg   signed [31:0] ac, bd, tmp;
    begin
      ac  = $signed(aI) * $signed(bI);
      bd  = $signed(aQ) * $signed(bQ);
      tmp = ac - bd;
      tmp = tmp >>> 7;
      cmul_re_q7 = sat9(tmp);
    end
  endfunction

  function signed [DWIDTH-1:0] cmul_im_q7;
    input signed [DWIDTH-1:0] aI, aQ, bI, bQ;
    reg   signed [31:0] ad, bc, tmp;
    begin
      ad  = $signed(aI) * $signed(bQ);
      bc  = $signed(aQ) * $signed(bI);
      tmp = ad + bc;
      tmp = tmp >>> 7;
      cmul_im_q7 = sat9(tmp);
    end
  endfunction

  // X/Z detect (barato)
  function is_unknown9;
    input [DWIDTH-1:0] v;
    begin
      is_unknown9 = (^v === 1'bx);
    end
  endfunction

  integer frames;
  integer err;

  reg signed [DWIDTH-1:0] expI, expQ;
  reg signed [DWIDTH-1:0] y0I, y0Q, y1I, y1Q;
  reg [1:0] check_mode; // solo para imprimir

  always @(posedge clk) begin
    if (rst) begin
      frames <= 0;
      err    <= 0;
    end else begin
      if (y_valid) begin
        // integridad básica
        if (is_unknown9(y_I) || is_unknown9(y_Q)) begin
          $display("[ERR] X/Z en salida y @t=%0t", $time);
          err <= err + 1;
        end

        // recomputo CMUL + SUM como el top
        y0I = cmul_re_q7(hb_curr_I, hb_curr_Q, i_W0_re, i_W0_im);
        y0Q = cmul_im_q7(hb_curr_I, hb_curr_Q, i_W0_re, i_W0_im);
        y1I = cmul_re_q7(hb_old_I,  hb_old_Q,  i_W1_re, i_W1_im);
        y1Q = cmul_im_q7(hb_old_I,  hb_old_Q,  i_W1_re, i_W1_im);

        expI = sat9($signed(y0I) + $signed(y1I));
        expQ = sat9($signed(y0Q) + $signed(y1Q));

        if (y_I !== expI || y_Q !== expQ) begin
          $display("[ERR] k=%0d y=(%0d,%0d) exp=(%0d,%0d) curr=(%0d,%0d) old=(%0d,%0d) W0=(%0d,%0d) W1=(%0d,%0d) @t=%0t",
                   y_k_idx, y_I, y_Q, expI, expQ,
                   hb_curr_I, hb_curr_Q, hb_old_I, hb_old_Q,
                   i_W0_re, i_W0_im, i_W1_re, i_W1_im, $time);
          err <= err + 1;
        end

        if (y_k_idx == (NFFT-1)) begin
          frames <= frames + 1;
          $display("[INFO] frame=%0d done (err=%0d) @t=%0t", frames+1, err, $time);
        end
      end
    end
  end

  task run_frames;
    input integer n;
    integer target;
    begin
      target = frames + n;
      wait(frames >= target);
    end
  endtask

  initial begin
    // init
    rst         = 1'b1;
    sigma_scale = 0;

    bypass_tx   = 1'b1;
    test_data_I = 9'sd100;
    test_data_Q = 9'sd0;

    i_W0_re = ONE_Q7;  i_W0_im = ZERO_Q7;
    i_W1_re = ONE_Q7;  i_W1_im = ZERO_Q7;

    repeat(20) @(posedge clk);
    rst = 1'b0;

    $display("=== INICIO SIM ===");
    $display("Reset liberado.");

    // -----------------------------------------
    // TEST 1A: y = curr  (W0=1 W1=0)
    // -----------------------------------------
    $display("--- TEST 1A: DC bypass, W0=1 W1=0 => y=curr ---");
    bypass_tx = 1'b1;
    i_W0_re = ONE_Q7;  i_W0_im = ZERO_Q7;
    i_W1_re = ZERO_Q7; i_W1_im = ZERO_Q7;
    run_frames(5);

    // -----------------------------------------
    // TEST 1B: y = old  (W0=0 W1=1)
    // -----------------------------------------
    $display("--- TEST 1B: DC bypass, W0=0 W1=1 => y=old ---");
    i_W0_re = ZERO_Q7; i_W0_im = ZERO_Q7;
    i_W1_re = ONE_Q7;  i_W1_im = ZERO_Q7;
    run_frames(5);

    // -----------------------------------------
    // TEST 1C: y = curr + old (W0=1 W1=1)
    // -----------------------------------------
    $display("--- TEST 1C: DC bypass, W0=1 W1=1 => y=curr+old ---");
    i_W0_re = ONE_Q7;  i_W0_im = ZERO_Q7;
    i_W1_re = ONE_Q7;  i_W1_im = ZERO_Q7;
    run_frames(5);

    // -----------------------------------------
    // TEST 1D: y = j*curr  (W0=j1 W1=0)
    //  j1 => (0, +128)  => yI=-currQ ; yQ=+currI
    // -----------------------------------------
    $display("--- TEST 1D: DC bypass, W0=j1 W1=0 => y = j*curr ---");
    i_W0_re = ZERO_Q7; i_W0_im = ONE_Q7;
    i_W1_re = ZERO_Q7; i_W1_im = ZERO_Q7;
    run_frames(5);

    // -----------------------------------------
    // TEST 2: TX real (bypass=0), deja W0=1 W1=1
    // -----------------------------------------
    $display("--- TEST 2: TX real (bypass_tx=0), W0=1 W1=1 ---");
    bypass_tx = 1'b0;
    i_W0_re = ONE_Q7;  i_W0_im = ZERO_Q7;
    i_W1_re = ONE_Q7;  i_W1_im = ZERO_Q7;
    run_frames(10);

    $display("=== FIN SIM === frames=%0d err=%0d", frames, err);
    $stop;
  end

endmodule
