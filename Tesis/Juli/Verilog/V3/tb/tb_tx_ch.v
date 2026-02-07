`timescale 1ns/1ps

module tb_tx;

  // ============================================================
  // PARAMS
  // ============================================================
  localparam integer NSYM        = 500;      // cantidad de simbolos a testear
  localparam integer TIMEOUT_CYC = 200000;
  localparam signed  [8:0] AMP   = 9'sd90;

  // seeds (no nulos)
  localparam [8:0] SEED_I = 9'h1A5;
  localparam [8:0] SEED_Q = 9'h12F;

  // ============================================================
  // CLK / RST
  // ============================================================
  reg clk, rst;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // ============================================================
  // DUT IO
  // ============================================================
  reg  en;

  wire prbsI_bit;
  wire prbsQ_bit;

  wire signed [8:0] sym_I;
  wire signed [8:0] sym_Q;

  // ============================================================
  // DUTs
  // ============================================================
  prbs9 u_prbsI (
    .clk  (clk),
    .rst  (rst),
    .en   (en),
    .seed (SEED_I),
    .bit  (prbsI_bit)
  );

  prbs9 u_prbsQ (
    .clk  (clk),
    .rst  (rst),
    .en   (en),
    .seed (SEED_Q),
    .bit  (prbsQ_bit)
  );

  qpsk_mapper u_map (
    .bit_I (prbsI_bit),
    .bit_Q (prbsQ_bit),
    .sym_I (sym_I),
    .sym_Q (sym_Q)
  );

  // ============================================================
  // REF MODEL (PRBS) para VM
  // ============================================================
  reg [8:0] sI_ref;
  reg [8:0] sQ_ref;

  wire fbI_ref = sI_ref[8] ^ sI_ref[4];
  wire fbQ_ref = sQ_ref[8] ^ sQ_ref[4];

  wire bitI_exp = sI_ref[8];
  wire bitQ_exp = sQ_ref[8];

  wire signed [8:0] symI_exp = (bitI_exp == 1'b0) ? +AMP : -AMP;
  wire signed [8:0] symQ_exp = (bitQ_exp == 1'b0) ? +AMP : -AMP;

  // ============================================================
  // SCOREBOARD
  // ============================================================
  integer i;
  integer err_cnt;

  // ============================================================
  // MAIN
  // ============================================================
  initial begin
    err_cnt = 0;
    en      = 1'b0;

    rst = 1'b1;
    repeat (5) @(posedge clk);
    rst = 1'b0;

    // init ref states igual que DUT
    sI_ref = (SEED_I != 9'd0) ? SEED_I : 9'h001;
    sQ_ref = (SEED_Q != 9'd0) ? SEED_Q : 9'h001;

    // avanzar y comparar
    for (i = 0; i < NSYM; i = i + 1) begin
      // en=1 por 1 ciclo
      @(negedge clk);
      en = 1'b1;

      @(posedge clk);
      // VM: comparar bits y simbolos en el mismo ciclo del enable
      if (prbsI_bit !== bitI_exp) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: PRBS_I mismatch i=%0d @t=%0t got=%b exp=%b", i, $time, prbsI_bit, bitI_exp);
      end
      if (prbsQ_bit !== bitQ_exp) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: PRBS_Q mismatch i=%0d @t=%0t got=%b exp=%b", i, $time, prbsQ_bit, bitQ_exp);
      end

      if (sym_I !== symI_exp) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: SYM_I mismatch i=%0d @t=%0t got=%0d exp=%0d", i, $time, $signed(sym_I), $signed(symI_exp));
      end
      if (sym_Q !== symQ_exp) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: SYM_Q mismatch i=%0d @t=%0t got=%0d exp=%0d", i, $time, $signed(sym_Q), $signed(symQ_exp));
      end

      // actualizar modelo ref (igual al always del PRBS)
      if (en) begin
        sI_ref <= {sI_ref[7:0], fbI_ref};
        sQ_ref <= {sQ_ref[7:0], fbQ_ref};
      end

      @(negedge clk);
      en = 1'b0;
    end

    if (err_cnt == 0)
      $display("TEST OK: PRBS9 + QPSK mapper (%0d simbolos) sin errores", NSYM);
    else
      $display("TEST FAIL: errores=%0d sobre %0d simbolos", err_cnt, NSYM);

    $finish;
  end

endmodule
