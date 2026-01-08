
module fft_ifft #(
  parameter integer NFFT            = 32,  // puntos (potencia de 2)
  parameter integer W               = 16,  // ancho 
  parameter integer FRAC            = 14,  // bits frac Q
  parameter integer SCALE_PER_STAGE = 1    // >>1 por etapa
)(
  input  wire                 i_clk,      // clock
  input  wire                 i_rst,      // reset

  input  wire                 i_start,    // inicia bloque
  input  wire                 i_ifft,     // 0: FFT, 1: IFFT

  input  wire                 i_valid,    // valida entrada
  input  wire signed [W-1:0]  i_x_re,     // x real
  input  wire signed [W-1:0]  i_x_im,     // x imag
  output wire                 o_ready,    // listo para cargar

  output reg                  o_valid,    // valida salida
  output reg  signed [W-1:0]  o_y_re,     // y real
  output reg  signed [W-1:0]  o_y_im,     // y imag
  output reg                  o_last      // ultimo sample
);

  localparam integer LOGN = $clog2(NFFT);
  localparam integer TW_W = $clog2(NFFT/2);

  // memoria
  reg signed [W-1:0] mem_re [0:NFFT-1];
  reg signed [W-1:0] mem_im [0:NFFT-1];

  // bit-reverse 
  function [LOGN-1:0] bit_reverse(input [LOGN-1:0] a);
    integer b;
    begin
      for (b = 0; b < LOGN; b = b + 1)
        bit_reverse[b] = a[LOGN-1-b];
    end
  endfunction

  // twiddle ROM
  wire [TW_W-1:0]     w_addr;   // addr twiddle
  wire signed [W-1:0] w_re_rom; // W real
  wire signed [W-1:0] w_im_rom; // W imag

  twiddle_rom #(
    .NFFT (NFFT),
    .W    (W),
    .FRAC (FRAC)
  ) u_twiddle_rom (
    .i_addr (w_addr),
    .o_re   (w_re_rom),
    .o_im   (w_im_rom)
  );

  // IFFT: conj(W) 
  wire signed [W-1:0] w_re = w_re_rom;
  wire signed [W-1:0] w_im = (i_ifft) ? -w_im_rom : w_im_rom;

  // FSM
  localparam [2:0]
    S_IDLE = 3'd0,
    S_LOAD = 3'd1,
    S_COMP = 3'd2,
    S_OUT  = 3'd3;

  reg [2:0] state;

  reg [LOGN:0] load_cnt; 
  reg [LOGN:0] out_cnt;  

  // radix-2
  reg [LOGN-1:0] stage;  // etapa 0..LOGN-1
  reg [LOGN-1:0] k_base; // base del grupo
  reg [LOGN-1:0] j;      // index dentro del grupo

  wire [LOGN-1:0] half_m = (1 << stage);       // m/2
  wire [LOGN-1:0] m      = (1 << (stage + 1)); // m

  // w_k = j * (N/m)
  wire [LOGN-1:0] step     = (NFFT >> (stage + 1));
  wire [LOGN-1:0] w_k_full = j * step;

  assign w_addr = w_k_full[TW_W-1:0];

  // indices de memoria
  wire [LOGN-1:0] idx_a = k_base + j;
  wire [LOGN-1:0] idx_b = k_base + j + half_m;

  // lecturas 
  wire signed [W-1:0] a_re = mem_re[idx_a];
  wire signed [W-1:0] a_im = mem_im[idx_a];
  wire signed [W-1:0] b_re = mem_re[idx_b];
  wire signed [W-1:0] b_im = mem_im[idx_b];

  // butterfly
  wire signed [W-1:0] y0_re;
  wire signed [W-1:0] y0_im;
  wire signed [W-1:0] y1_re;
  wire signed [W-1:0] y1_im;

  butterfly #(
    .W        (W),
    .FRAC     (FRAC),
    .SCALE_EN (SCALE_PER_STAGE)
  ) u_bfly (
    .i_a_re (a_re),
    .i_a_im (a_im),
    .i_b_re (b_re),
    .i_b_im (b_im),
    .i_w_re (w_re),
    .i_w_im (w_im),
    .o_y0_re(y0_re),
    .o_y0_im(y0_im),
    .o_y1_re(y1_re),
    .o_y1_im(y1_im)
  );

  assign o_ready = (state == S_LOAD);

  
  always @(posedge i_clk) begin
    if (i_rst) begin
      state    <= S_IDLE;
      load_cnt <= 0;
      out_cnt  <= 0;
      stage    <= 0;
      k_base   <= 0;
      j        <= 0;
      o_valid  <= 0;
      o_last   <= 0;
      o_y_re   <= 0;
      o_y_im   <= 0;
    end else begin
      o_valid <= 0;
      o_last  <= 0;

      case (state)

        S_IDLE: begin
          if (i_start) begin
            load_cnt <= 0;
            state    <= S_LOAD;
          end
        end

        // carga NFFT samples en bit-reverse
        S_LOAD: begin
          if (i_valid) begin
            mem_re[ bit_reverse(load_cnt[LOGN-1:0]) ] <= i_x_re;
            mem_im[ bit_reverse(load_cnt[LOGN-1:0]) ] <= i_x_im;

            load_cnt <= load_cnt + 1;

            if (load_cnt == NFFT-1) begin
              stage  <= 0;
              k_base <= 0;
              j      <= 0;
              state  <= S_COMP;
            end
          end
        end

        // computa 1 butterfly por ciclo
        S_COMP: begin
          mem_re[idx_a] <= y0_re;
          mem_im[idx_a] <= y0_im;
          mem_re[idx_b] <= y1_re;
          mem_im[idx_b] <= y1_im;

          if (j == half_m-1) begin
            j <= 0;

            if (k_base == NFFT - m) begin
              k_base <= 0;

              if (stage == LOGN-1) begin
                out_cnt <= 0;
                state   <= S_OUT;
              end else begin
                stage <= stage + 1;
              end

            end else begin
              k_base <= k_base + m;
            end

          end else begin
            j <= j + 1;
          end
        end

        // saca NFFT outputs en orden natural
        S_OUT: begin
          o_valid <= 1;
          o_y_re  <= mem_re[out_cnt[LOGN-1:0]];
          o_y_im  <= mem_im[out_cnt[LOGN-1:0]];

          if (out_cnt == NFFT-1) begin
            o_last <= 1;
            state  <= S_IDLE;
          end

          out_cnt <= out_cnt + 1;
        end

        default: state <= S_IDLE;

      endcase
    end
  end

endmodule
