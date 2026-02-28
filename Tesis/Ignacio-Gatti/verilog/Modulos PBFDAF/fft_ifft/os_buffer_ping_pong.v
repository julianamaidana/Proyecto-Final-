module os_buffer #(
  parameter integer N  = 16, // PART_N
  parameter integer WN = 9   // FX_NARROW S(9,7)
)(
  input  wire                  i_clk,
  input  wire                  i_rst,

  input  wire                  i_valid,
  input  wire                  i_ce_in,   // CE del "mundo lento" (1 cada 2 clocks)
  input  wire signed [WN-1:0]  i_xI,
  input  wire signed [WN-1:0]  i_xQ,

  output reg                   o_overflow,  // sticky: 1 si alguna vez se dropeo un sample

  output reg                   o_fft_start, // pulso 1 ciclo (inicio de frame)
  output reg                   o_fft_valid, // 1 mientras envia 2N samples
  output reg  signed [WN-1:0]  o_fft_xI,
  output reg  signed [WN-1:0]  o_fft_xQ
);

  // log2
  function integer clog2;
    input integer value;
    integer i;
    begin
      clog2 = 0;
      for (i = value - 1; i > 0; i = i >> 1)
        clog2 = clog2 + 1;
    end
  endfunction

  localparam integer CNTW = (N <= 1) ? 1 : clog2(N);
  localparam integer IDXW = clog2(2*N);

  integer j;

  // overlap (bloque previo confirmado)
  reg signed [WN-1:0] overlapI [0:N-1];
  reg signed [WN-1:0] overlapQ [0:N-1];

  // ping-pong buffers para el bloque "new"
  reg signed [WN-1:0] newAI [0:N-1];
  reg signed [WN-1:0] newAQ [0:N-1];
  reg signed [WN-1:0] newBI [0:N-1];
  reg signed [WN-1:0] newBQ [0:N-1];

  // flags de bancos listos
  reg bankA_full;
  reg bankB_full;

  // control de llenado
  reg            fill_bank; // 0->A, 1->B
  reg [CNTW-1:0] fill_cnt;

  // control de envío
  reg            sending;
  reg            send_bank; // 0->A, 1->B
  reg [IDXW-1:0] send_idx;

  // -----------------------------
  // HOLD 1-deep (skid buffer)
  // -----------------------------
  reg hold_v;
  reg signed [WN-1:0] hold_I;
  reg signed [WN-1:0] hold_Q;

  // cuando NO hay ready: si el banco de llenado está full => no hay dónde guardar
  wire fill_blocked = (fill_bank == 1'b0) ? bankA_full : bankB_full;

  // Entrada "real" solo con CE
  wire in_raw = i_valid && i_ce_in;

  // Si hay hold, se prioriza vaciar hold antes que aceptar una nueva muestra
  wire use_hold = hold_v;

  // Hay algo para intentar escribir este ciclo si:
  // - o bien hay hold pendiente
  // - o bien llegó una muestra nueva (in_raw)
  wire src_fire = use_hold || in_raw;

  // Se puede escribir si el banco de llenado no está bloqueado
  wire can_write = !fill_blocked;

  // Escribimos si hay fuente y hay lugar
  wire do_write = src_fire && can_write;

  // Dato que efectivamente se escribe
  wire signed [WN-1:0] wI = use_hold ? hold_I : i_xI;
  wire signed [WN-1:0] wQ = use_hold ? hold_Q : i_xQ;

  // mux del frame durante SEND
  wire signed [WN-1:0] send_newI =
    (send_bank == 1'b0) ? newAI[send_idx - N] : newBI[send_idx - N];
  wire signed [WN-1:0] send_newQ =
    (send_bank == 1'b0) ? newAQ[send_idx - N] : newBQ[send_idx - N];

  wire signed [WN-1:0] blkI = (send_idx < N) ? overlapI[send_idx] : send_newI;
  wire signed [WN-1:0] blkQ = (send_idx < N) ? overlapQ[send_idx] : send_newQ;

  // prioridad simple para elegir banco a enviar: A si está lleno, si no B
  wire have_block = bankA_full | bankB_full;
  wire pick_A     = bankA_full;

  always @(posedge i_clk) begin
    if (i_rst) begin
      o_overflow  <= 1'b0;

      o_fft_start <= 1'b0;
      o_fft_valid <= 1'b0;
      o_fft_xI    <= {WN{1'b0}};
      o_fft_xQ    <= {WN{1'b0}};

      bankA_full  <= 1'b0;
      bankB_full  <= 1'b0;

      fill_bank   <= 1'b0;
      fill_cnt    <= {CNTW{1'b0}};

      sending     <= 1'b0;
      send_bank   <= 1'b0;
      send_idx    <= {IDXW{1'b0}};

      hold_v      <= 1'b0;
      hold_I      <= {WN{1'b0}};
      hold_Q      <= {WN{1'b0}};

      for (j = 0; j < N; j = j + 1) begin
        overlapI[j] <= {WN{1'b0}};
        overlapQ[j] <= {WN{1'b0}};
        newAI[j]    <= {WN{1'b0}};
        newAQ[j]    <= {WN{1'b0}};
        newBI[j]    <= {WN{1'b0}};
        newBQ[j]    <= {WN{1'b0}};
      end

    end else begin
      o_fft_start <= 1'b0;

      // ------------------------------------------------------------
      // HOLD capture:
      // Si llegó una muestra nueva (in_raw) y no hay lugar para escribir
      // y NO tenemos hold ocupado -> guardar en hold.
      // Si hold ya estaba ocupado y vuelve a llegar otra muestra bloqueada,
      // entonces sí: overflow (dropeo real).
      // ------------------------------------------------------------
      if (in_raw && fill_blocked) begin
        if (!hold_v) begin
          hold_v <= 1'b1;
          hold_I <= i_xI;
          hold_Q <= i_xQ;
        end else begin
          // hold ocupado y otra muestra también bloqueada => drop real
          o_overflow <= 1'b1;
        end
      end

      // ------------------------------------------------------------
      // 1) FILL: si hay fuente (hold o muestra nueva) y hay lugar, escribir
      // ------------------------------------------------------------
      if (do_write) begin
        // Si estamos consumiendo hold, liberarlo
        if (hold_v) begin
          hold_v <= 1'b0;
        end

        if (fill_bank == 1'b0) begin
          newAI[fill_cnt] <= wI;
          newAQ[fill_cnt] <= wQ;
        end else begin
          newBI[fill_cnt] <= wI;
          newBQ[fill_cnt] <= wQ;
        end

        if (fill_cnt == N-1) begin
          fill_cnt <= {CNTW{1'b0}};

          // marcar banco como lleno
          if (fill_bank == 1'b0) bankA_full <= 1'b1;
          else                   bankB_full <= 1'b1;

          // conmutar al otro banco
          fill_bank <= ~fill_bank;
        end else begin
          fill_cnt <= fill_cnt + 1'b1;
        end
      end

      // ------------------------------------------------------------
      // 2) SEND: emite 2N = [overlap | newX]
      // ------------------------------------------------------------
      if (!sending) begin
        o_fft_valid <= 1'b0;

        if (have_block) begin
          send_bank   <= pick_A ? 1'b0 : 1'b1;
          send_idx    <= {IDXW{1'b0}};
          sending     <= 1'b1;
          o_fft_start <= 1'b1;
        end
      end else begin
        o_fft_valid <= 1'b1;
        o_fft_xI    <= blkI;
        o_fft_xQ    <= blkQ;

        if (send_idx == (2*N - 1)) begin
          sending  <= 1'b0;
          send_idx <= {IDXW{1'b0}};

          // actualizar overlap <= new[send_bank]
          for (j = 0; j < N; j = j + 1) begin
            overlapI[j] <= (send_bank == 1'b0) ? newAI[j] : newBI[j];
            overlapQ[j] <= (send_bank == 1'b0) ? newAQ[j] : newBQ[j];
          end

          // liberar banco enviado
          if (send_bank == 1'b0) bankA_full <= 1'b0;
          else                   bankB_full <= 1'b0;

        end else begin
          send_idx <= send_idx + 1'b1;
        end
      end
    end
  end

endmodule
