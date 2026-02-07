`timescale 1ns/1ps

module tb_sat_trunc;

  // FX_WIDE  (17,10) -> FX_NARROW (9,7)
  localparam integer NB_XI  = 17;
  localparam integer NBF_XI = 10;
  localparam integer NB_XO  = 9;
  localparam integer NBF_XO = 7;

  localparam integer K = (NBF_XI - NBF_XO); // 3

  // -------------------------
  // DUT I/O (combinacional)
  // -------------------------
  reg  signed [NB_XI-1:0] i_data;
  wire signed [NB_XO-1:0] o_data;

  
  sat_trunc #(
    .NB_XI (NB_XI),
    .NBF_XI(NBF_XI),
    .NB_XO (NB_XO),
    .NBF_XO(NBF_XO),
    .ROUND (1)
  ) dut (
    .i_data(i_data),
    .o_data(o_data)
  );

  // -------------------------
  // Golden: round_even + saturate
  // -------------------------
  function signed [NB_XO-1:0] golden_round_even_sat;
    input signed [NB_XI-1:0] x;

    reg signed [NB_XI-1:0] y;         // x >> K
    reg guard;
    reg sticky;
    reg inc;

    reg signed [NB_XI-1:0] y_round;

    // saturación en "raw" de NB_XO bits signed
    reg signed [NB_XI-1:0] max_raw_ext;
    reg signed [NB_XI-1:0] min_raw_ext;

    begin
      // shift aritmetico para igualar fracción (Q10 -> Q7)
      y = (K > 0) ? (x >>> K) : x;

      // round-to-even cuando K>0:
      // guard  = bit K-1 (MSB de lo descartado)
      // sticky = OR bits K-2..0
      // inc = guard & (sticky | (LSB del kept)) ; LSB del kept = y[0]
      if (K > 0) begin
        guard  = x[K-1];
        sticky = (K > 1) ? (|x[K-2:0]) : 1'b0;
        inc    = guard & (sticky | y[0]);
      end else begin
        inc    = 1'b0;
      end

      y_round = y + (inc ? 1 : 0);

      // saturación signed a NB_XO bits
      // max_raw =  2^(NB_XO-1)-1 ; min_raw = -2^(NB_XO-1)
      max_raw_ext =  ( (1 <<< (NB_XO-1)) - 1 );
      min_raw_ext = -(  1 <<< (NB_XO-1)      );

      if (y_round > max_raw_ext)
        golden_round_even_sat = max_raw_ext[NB_XO-1:0];
      else if (y_round < min_raw_ext)
        golden_round_even_sat = min_raw_ext[NB_XO-1:0];
      else
        golden_round_even_sat = y_round[NB_XO-1:0];
    end
  endfunction

  integer err_cnt;

  task apply_check;
    input signed [NB_XI-1:0] x;
    reg signed [NB_XO-1:0] y_exp;
    begin
      i_data = x;
      #1;
      y_exp = golden_round_even_sat(x);

      if (o_data !== y_exp) begin
        err_cnt = err_cnt + 1;
        $display("ERROR: in=%0d (0x%h)  got=%0d (0x%h)  exp=%0d (0x%h)",
                 $signed(i_data), i_data, $signed(o_data), o_data, $signed(y_exp), y_exp);
      end
    end
  endtask

  // Helpers para armar patrones críticos de rounding
  // Tie exacto (descartado = 100...0) para K=3 => bits[2:0] = 100
  function signed [NB_XI-1:0] with_low3;
    input signed [NB_XI-1:0] base; // múltiplo de 8 idealmente
    input [2:0] low3;
    begin
      with_low3 = (base & ~17'sd7) | low3; // fuerza bits[2:0]
    end
  endfunction

  integer n;
  reg signed [NB_XI-1:0] x;

  initial begin
    err_cnt = 0;
    i_data  = 0;

    // ============================================================
    // CASOS CRÍTICOS DE ROUND_EVEN (K=3)
    // ============================================================

    // 1) Tie exacto con LSB del kept = 0  -> NO incrementa
    //    x = 0b....000 + 0b100 = 4  => y=0, guard=1, sticky=0, y[0]=0 => inc=0
    apply_check(17'sd4);

    // 2) Tie exacto con LSB del kept = 1  -> SI incrementa (to even)
    //    x = 12 = 0b1100 -> y=1, guard=1 sticky=0 y[0]=1 => inc=1 => y_round=2
    apply_check(17'sd12);

    // 3) Justo abajo del tie: 0b011 (guard=0) -> NO incrementa
    apply_check(17'sd3);

    // 4) Justo arriba del tie: 0b101 (guard=1 sticky=1) -> SI incrementa
    apply_check(17'sd5);

    // ============================================================
    // SATURACIÓN (NB_XO=9 => raw max=255, min=-256)
    // En Q7: max=255 => en Q10 equivalente ideal ~255<<3=2040
    // ============================================================

    // 5) En el máximo exacto representable (sin saturar)
    apply_check(17'sd2040); // -> y=255

    // 6) Apenas por encima (debería saturar a 255)
    apply_check(17'sd2048); // -> y=256 (overflow) => sat 255

    // 7) Caso donde el rounding empuja al overflow (2044 tiene low3=100 => tie)
    //    2044 >>> 3 = 255, tie con y[0]=1 => inc => 256 => sat 255
    apply_check(17'sd2044);

    // 8) Mínimo exacto (sin saturar): -256 << 3 = -2048
    apply_check(-17'sd2048); // -> y=-256

    // 9) Apenas más negativo (satura a -256)
    apply_check(-17'sd2056); // -> y=-257 => sat -256

    // ============================================================
    // RANDOM (VM)
    // ============================================================
    for (n = 0; n < 500; n = n + 1) begin
      x = $random;                 // 32b, se trunca a 17b signed
      apply_check(x[NB_XI-1:0]);
    end

    if (err_cnt == 0) $display("TEST OK: sin errores (round_even + saturate)");
    else              $display("TEST FAIL: errores=%0d", err_cnt);

    $finish;
  end

endmodule
