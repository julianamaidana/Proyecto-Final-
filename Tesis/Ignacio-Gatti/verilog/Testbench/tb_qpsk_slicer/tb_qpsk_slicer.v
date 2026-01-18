`timescale 1ns/1ps

module tb_qpsk_slicer;

  // =========================
  // Parámetros (Verilog-2001)
  // =========================
  parameter integer W    = 16;
  parameter integer FRAC = 14;
  parameter signed [W-1:0] TH = 0;

  // +1.0 en Q(FRAC)
  localparam signed [W-1:0] AMP   = (1 <<< FRAC);
  // Entrada de prueba: 0.5 (para no saturar, solo prueba de signo)
  localparam signed [W-1:0] INMAG = (1 <<< (FRAC-1));

  parameter integer NBLOCKS = 3;
  parameter integer GAP_CYC = 5;

  // =========================
  // Clock / Reset
  // =========================
  reg i_clk;
  reg i_rst;

  // =========================
  // Entradas DUT
  // =========================
  reg                 i_valid;
  reg                 i_first;
  reg                 i_last;
  reg signed [W-1:0]  i_y_re;
  reg signed [W-1:0]  i_y_im;

  // =========================
  // Salidas DUT
  // =========================
  wire                o_valid;
  wire                o_first;
  wire                o_last;
  wire                o_bI_hat;
  wire                o_bQ_hat;
  wire signed [W-1:0] o_yhat_re;
  wire signed [W-1:0] o_yhat_im;

  // =========================
  // DUT
  // =========================
  qpsk_slicer #(
    .W(W),
    .FRAC(FRAC),
    .TH(TH)
  ) dut (
    .i_clk     (i_clk),
    .i_rst     (i_rst),
    .i_valid   (i_valid),
    .i_first   (i_first),
    .i_last    (i_last),
    .i_y_re    (i_y_re),
    .i_y_im    (i_y_im),
    .o_valid   (o_valid),
    .o_first   (o_first),
    .o_last    (o_last),
    .o_bI_hat  (o_bI_hat),
    .o_bQ_hat  (o_bQ_hat),
    .o_yhat_re (o_yhat_re),
    .o_yhat_im (o_yhat_im)
  );

  // =========================
  // Clock
  // =========================
  initial i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  // =========================
  // Task: gap (i_valid=0)
  // =========================
  task do_gap;
    input integer cycles;
    integer k;
    begin
      i_valid = 0;
      i_first = 0;
      i_last  = 0;
      i_y_re  = 0;
      i_y_im  = 0;
      for (k = 0; k < cycles; k = k + 1)
        @(posedge i_clk);
    end
  endtask

  // =========================
  // Task: enviar símbolo y chequear
  // exp_bI/exp_bQ: 1 si negativo, 0 si positivo (según TH)
  // =========================
  task send_and_check;
    input signed [W-1:0] re;
    input signed [W-1:0] im;
    input exp_bI;
    input exp_bQ;
    input is_first;
    input is_last;

    reg signed [W-1:0] exp_yhat_re;
    reg signed [W-1:0] exp_yhat_im;
    begin
      exp_yhat_re = (exp_bI ? -AMP : AMP);
      exp_yhat_im = (exp_bQ ? -AMP : AMP);

      // aplicar estímulo
      i_valid = 1;
      i_first = is_first;
      i_last  = is_last;
      i_y_re  = re;
      i_y_im  = im;

      @(posedge i_clk);

      // ===== CHECKS =====
      if (o_valid !== 1'b1) begin
        $display("ERROR: o_valid=0 con i_valid=1");
        $fatal;
      end

      if (o_bI_hat !== exp_bI) begin
        $display("ERROR: o_bI_hat=%0d esperado=%0d (re=%0d)", o_bI_hat, exp_bI, re);
        $fatal;
      end

      if (o_bQ_hat !== exp_bQ) begin
        $display("ERROR: o_bQ_hat=%0d esperado=%0d (im=%0d)", o_bQ_hat, exp_bQ, im);
        $fatal;
      end

      if (o_yhat_re !== exp_yhat_re) begin
        $display("ERROR: o_yhat_re=%0d esperado=%0d", o_yhat_re, exp_yhat_re);
        $fatal;
      end

      if (o_yhat_im !== exp_yhat_im) begin
        $display("ERROR: o_yhat_im=%0d esperado=%0d", o_yhat_im, exp_yhat_im);
        $fatal;
      end

      if (o_first !== is_first) begin
        $display("ERROR: o_first=%0d esperado=%0d", o_first, is_first);
        $fatal;
      end

      if (o_last !== is_last) begin
        $display("ERROR: o_last=%0d esperado=%0d", o_last, is_last);
        $fatal;
      end

      // bajar valid (para que quede prolijo entre símbolos)
      i_valid = 0;
      i_first = 0;
      i_last  = 0;
    end
  endtask

  // =========================
  // MAIN
  // =========================
  integer blk;
  initial begin
    // init
    i_rst   = 1;
    i_valid = 0;
    i_first = 0;
    i_last  = 0;
    i_y_re  = 0;
    i_y_im  = 0;

    // reset
    repeat(4) @(posedge i_clk);
    i_rst = 0;

    // N bloques, cada uno con 4 símbolos (4 cuadrantes)
    for (blk = 0; blk < NBLOCKS; blk = blk + 1) begin
      do_gap(GAP_CYC);

      // (++), (-+), (--), (+-)
      send_and_check(+INMAG, +INMAG, 0, 0, 1, 0);
      send_and_check(-INMAG, +INMAG, 1, 0, 0, 0);
      send_and_check(-INMAG, -INMAG, 1, 1, 0, 0);
      send_and_check(+INMAG, -INMAG, 0, 1, 0, 1);
    end

    do_gap(10);
    $display("TB OK: qpsk_slicer pasó %0d bloques correctamente.", NBLOCKS);
    $finish;
  end

endmodule
