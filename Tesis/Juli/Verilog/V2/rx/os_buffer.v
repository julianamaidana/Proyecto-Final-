module os_buffer #(
  parameter integer N  = 16, // PART_N
  parameter integer WN = 9   // FX_NARROW S(9,7)
)(
  input  wire                  i_clk,
  input  wire                  i_rst,
  input  wire                  i_valid,
  input  wire signed [WN-1:0]  i_xI,
  input  wire signed [WN-1:0]  i_xQ,

  output wire                  o_in_ready,   // listo para recibir

  output reg                   o_fft_start,  // pulso 1 ciclo, antes del 1er sample
  output reg                   o_fft_valid,  // 1 mientras envia 2N samples
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

  // buffers: overlap (bloque previo) y new (bloque actual)
  reg signed [WN-1:0] overlapI [0:N-1];
  reg signed [WN-1:0] overlapQ [0:N-1];
  reg signed [WN-1:0] newI     [0:N-1];
  reg signed [WN-1:0] newQ     [0:N-1];

  reg [CNTW-1:0] cnt;        // cuenta samples nuevos
  reg [IDXW-1:0] send_idx;   // cuenta envio

  integer j;

  // FSM: junta N, dsp envia 2N = [overlap | new]
  localparam [1:0] S_COLLECT = 2'd0;
  localparam [1:0] S_SEND    = 2'd1;

  reg [1:0] state;

  assign o_in_ready = (state == S_COLLECT);

  // mux de bloque durante SEND
  wire signed [WN-1:0] blkI = (send_idx < N) ? overlapI[send_idx] : newI[send_idx - N];
  wire signed [WN-1:0] blkQ = (send_idx < N) ? overlapQ[send_idx] : newQ[send_idx - N];

  always @(posedge i_clk) begin
    if (i_rst) begin
      state       <= S_COLLECT;
      cnt         <= {CNTW{1'b0}};
      send_idx    <= {IDXW{1'b0}};
      o_fft_start <= 1'b0;
      o_fft_valid <= 1'b0;
      o_fft_xI    <= {WN{1'b0}};
      o_fft_xQ    <= {WN{1'b0}};

      for (j = 0; j < N; j = j + 1) begin
        overlapI[j] <= {WN{1'b0}};
        overlapQ[j] <= {WN{1'b0}};
        newI[j]     <= {WN{1'b0}};
        newQ[j]     <= {WN{1'b0}};
      end

    end else begin
      o_fft_start <= 1'b0;

      case (state)

        S_COLLECT: begin
          o_fft_valid <= 1'b0;

          if (i_valid) begin
            newI[cnt] <= i_xI;
            newQ[cnt] <= i_xQ;

            if (cnt == N-1) begin
              cnt         <= {CNTW{1'b0}};
              send_idx    <= {IDXW{1'b0}};
              o_fft_start <= 1'b1;      // pulso antes del 1er valid
              state       <= S_SEND;
            end else begin
              cnt <= cnt + 1'b1;
            end
          end
        end

        S_SEND: begin
          // FIX: mantener valid alto durante TODO el envio (2N ciclos).
          // NO bajar valid en el mismo ciclo del ultimo sample, porque se pierde idx=2N-1.
          o_fft_valid <= 1'b1;
          o_fft_xI    <= blkI;
          o_fft_xQ    <= blkQ;

          if (send_idx == (2*N - 1)) begin
            send_idx <= {IDXW{1'b0}};

            for (j = 0; j < N; j = j + 1) begin
              overlapI[j] <= newI[j];
              overlapQ[j] <= newQ[j];
            end

            state <= S_COLLECT; // en COLLECT o_fft_valid se baja
          end else begin
            send_idx <= send_idx + 1'b1;
          end
        end

        default: state <= S_COLLECT;

      endcase
    end
  end

endmodule
