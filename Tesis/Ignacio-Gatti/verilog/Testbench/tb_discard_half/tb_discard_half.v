`timescale 1ns/1ps

module tb_discard_half;

  // =========================
  // Parámetros
  // =========================
  parameter W       = 16;
  parameter NFFT    = 32;
  parameter DISCARD = 16;
  parameter NBLOCKS = 5;
  parameter GAP_CYC = 7;

  // =========================
  // Clock / Reset
  // =========================
  reg i_clk;
  reg i_rst;

  // =========================
  // Entradas DUT
  // =========================
  reg                 i_valid;
  reg signed [W-1:0]  i_y_re;
  reg signed [W-1:0]  i_y_im;
  reg                 i_last;

  // =========================
  // Salidas DUT
  // =========================
  wire                o_valid;
  wire signed [W-1:0] o_y_re;
  wire signed [W-1:0] o_y_im;
  wire                o_first;
  wire                o_last;
  wire [clog2(NFFT)-1:0] o_idx;

  // =========================
  // DUT
  // =========================
  discard_half #(
    .W(W),
    .NFFT(NFFT),
    .DISCARD(DISCARD)
  ) dut (
    .i_clk   (i_clk),
    .i_rst   (i_rst),
    .i_valid (i_valid),
    .i_y_re  (i_y_re),
    .i_y_im  (i_y_im),
    .i_last  (i_last),
    .o_valid (o_valid),
    .o_y_re  (o_y_re),
    .o_y_im  (o_y_im),
    .o_first (o_first),
    .o_last  (o_last),
    .o_idx   (o_idx)
  );

  // =========================
  // Clock gen
  // =========================
  initial i_clk = 0;
  always #5 i_clk = ~i_clk;

  // =========================
  // Variables para checks
  // =========================
  integer blk;
  integer n;
  integer base_bad;
  integer base_good;

  // =========================
  // Función clog2 (Verilog)
  // =========================
  function integer clog2;
    input integer value;
    integer i;
    begin
      clog2 = 0;
      for (i = value-1; i > 0; i = i >> 1)
        clog2 = clog2 + 1;
    end
  endfunction

  // =========================
  // Task: gap (i_valid=0)
  // =========================
  task do_gap;
    input integer cycles;
    integer k;
    begin
      i_valid = 0;
      i_last  = 0;
      i_y_re  = 0;
      i_y_im  = 0;
      for (k = 0; k < cycles; k = k + 1)
        @(posedge i_clk);
    end
  endtask

  // =========================
  // Task: enviar 1 bloque
  // - Mitad mala: base_bad + 0..(DISCARD-1)
  // - Mitad buena: base_good + 0..(NFFT-DISCARD-1)
  // Esperado: salida SOLO mitad buena
  // =========================
  task send_block;
    input integer blk_id;
    integer idx_in;
    integer expected_re;
    integer expected_im;
    begin
      base_bad  = blk_id * 100;
      base_good = 1000 + blk_id * 100;

      i_valid = 1;
      for (idx_in = 0; idx_in < NFFT; idx_in = idx_in + 1) begin

        if (idx_in < DISCARD) begin
          i_y_re = base_bad + idx_in;
          i_y_im = -(base_bad + idx_in);
        end else begin
          i_y_re = base_good + (idx_in - DISCARD);
          i_y_im = -(base_good + (idx_in - DISCARD));
        end

        i_last = (idx_in == (NFFT-1));

        @(posedge i_clk);

        // =========================
        // CHECKS
        // =========================
        if (idx_in < DISCARD) begin
          // mitad mala: o_valid debe ser 0
          if (o_valid !== 1'b0) begin
            $display("ERROR BLK=%0d idx_in=%0d: o_valid debería ser 0 en mitad mala. o_y_re=%0d",
                     blk_id, idx_in, o_y_re);
            $fatal;
          end
        end else begin
          // mitad buena: o_valid debe ser 1 y data debe coincidir
          if (o_valid !== 1'b1) begin
            $display("ERROR BLK=%0d idx_in=%0d: o_valid debería ser 1 en mitad buena.",
                     blk_id, idx_in);
            $fatal;
          end

          expected_re = base_good + (idx_in - DISCARD);
          expected_im = -(base_good + (idx_in - DISCARD));

          if (o_y_re !== expected_re) begin
            $display("ERROR BLK=%0d idx_in=%0d: o_y_re=%0d esperado=%0d",
                     blk_id, idx_in, o_y_re, expected_re);
            $fatal;
          end

          if (o_y_im !== expected_im) begin
            $display("ERROR BLK=%0d idx_in=%0d: o_y_im=%0d esperado=%0d",
                     blk_id, idx_in, o_y_im, expected_im);
            $fatal;
          end

          // o_idx debe ser idx_in - DISCARD
          if (o_idx !== (idx_in - DISCARD)) begin
            $display("ERROR BLK=%0d idx_in=%0d: o_idx=%0d esperado=%0d",
                     blk_id, idx_in, o_idx, (idx_in - DISCARD));
            $fatal;
          end

          // o_first solo cuando idx_in == DISCARD
          if ((idx_in == DISCARD) && (o_first !== 1'b1)) begin
            $display("ERROR BLK=%0d: faltó o_first en primer sample bueno", blk_id);
            $fatal;
          end
          if ((idx_in != DISCARD) && (o_first === 1'b1)) begin
            $display("ERROR BLK=%0d: o_first apareció fuera del primer sample bueno (idx_in=%0d)",
                     blk_id, idx_in);
            $fatal;
          end

          // o_last solo cuando i_last=1 (último del bloque, cae en mitad buena)
          if (i_last && (o_last !== 1'b1)) begin
            $display("ERROR BLK=%0d: faltó o_last cuando i_last=1", blk_id);
            $fatal;
          end
          if (!i_last && (o_last === 1'b1)) begin
            $display("ERROR BLK=%0d: o_last apareció antes del fin de bloque (idx_in=%0d)",
                     blk_id, idx_in);
            $fatal;
          end
        end
      end

      // bajamos valid al terminar bloque
      i_valid = 0;
      i_last  = 0;
      i_y_re  = 0;
      i_y_im  = 0;
    end
  endtask

  // =========================
  // Main
  // =========================
  initial begin
    // init
    i_rst   = 1;
    i_valid = 0;
    i_last  = 0;
    i_y_re  = 0;
    i_y_im  = 0;

    // reset
    repeat(4) @(posedge i_clk);
    i_rst = 0;

    // varios bloques con gaps
    for (blk = 0; blk < NBLOCKS; blk = blk + 1) begin
      do_gap(GAP_CYC);
      send_block(blk);
    end

    do_gap(10);
    $display("TB OK: discard_half pasó %0d bloques correctamente.", NBLOCKS);
    $finish;
  end

endmodule
