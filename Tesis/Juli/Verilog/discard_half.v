module discard_half #(
  parameter integer NFFT    = 32,
  parameter integer W       = 16,
  parameter integer DISCARD = (NFFT/2)   // para OS 2N: descarto N
)(
  input  wire                 i_clk,
  input  wire                 i_rst,

  input  wire                 i_valid,
  input  wire signed [W-1:0]  i_y_re,
  input  wire signed [W-1:0]  i_y_im,
  input  wire                 i_last,   // último sample del bloque NFFT

  output reg                  o_valid,
  output reg signed [W-1:0]   o_y_re,
  output reg signed [W-1:0]   o_y_im,
  output reg                  o_first,  // 1 ciclo en la primera muestra buena
  output reg                  o_last,   // 1 ciclo en la última muestra buena
  output reg [$clog2(NFFT)-1:0] o_idx   // 0..(NFFT-DISCARD-1), útil para debug
);

  localparam integer IDXW = $clog2(NFFT);
  reg [IDXW-1:0] idx; // 0..NFFT-1 durante la ráfaga

  always @(posedge i_clk) begin
    if (i_rst) begin
      idx     <= {IDXW{1'b0}};
      o_valid <= 1'b0;
      o_first <= 1'b0;
      o_last  <= 1'b0;
      o_y_re  <= {W{1'b0}};
      o_y_im  <= {W{1'b0}};
      o_idx   <= {IDXW{1'b0}};
    end else begin
      // defaults
      o_valid <= 1'b0;
      o_first <= 1'b0;
      o_last  <= 1'b0;

      if (!i_valid) begin
        // Entre ráfagas, se prepara para el próximo bloque
        idx <= {IDXW{1'b0}};
      end else begin
        // idx es el índice de ESTE sample
        if (idx >= DISCARD) begin
          o_valid <= 1'b1;
          o_y_re  <= i_y_re;
          o_y_im  <= i_y_im;

          o_idx   <= idx - DISCARD;

          if (idx == DISCARD) o_first <= 1'b1;
          if (i_last)         o_last  <= 1'b1; // el último siempre cae en la mitad “buena”
        end

        // avanza el contador para el próximo ciclo de la ráfaga
        if (i_last) idx <= {IDXW{1'b0}};
        else        idx <= idx + 1'b1;
      end
    end
  end

endmodule
