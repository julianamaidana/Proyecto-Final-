`timescale 1ns/1ps
`default_nettype none

// ============================================================
// zero_pad_error  v2
//
// Propósito:
//   Primera etapa del loop LMS del PBFDAF.
//   Convierte el bloque de error e_blk (N muestras del slicer)
//   en un frame de 2N muestras listo para FFT_ERROR:
//
//     Salida = [ 0, 0, ..., 0,  e[0], e[1], ..., e[N-1] ]
//               ←   N ceros  →  ←     N errores      →
//
// Timing real (compatible con el slicer):
//
//   El slicer envía N errores con i_valid durante N ciclos.
//   El zero_pad los BUFFERIZA internamente y luego emite:
//     - N ceros  (primera mitad del frame de salida)
//     - N errores del buffer (segunda mitad)
//
//   FSM de 3 estados:
//     RECV  (N ciclos): recibe i_valid, guarda en buffer RAM
//     ZEROS (N ciclos): emite N ceros con o_valid
//     ERROR (N ciclos): emite N errores del buffer con o_valid
//
//   Latencia: N ciclos de recepción antes de emitir el frame
//
// Parámetros:
//   NB_W  = ancho de dato (9 bits, Q9.7)
//   NFFT  = 2N = tamaño del frame FFT de salida (32)
// ============================================================

module zero_pad_error #(
    parameter integer NB_W = 9,
    parameter integer NFFT = 32
)(
    input  wire                    clk,
    input  wire                    rst,

    // --- Entrada: e_blk del slicer (N muestras) ---
    input  wire                    i_valid,
    input  wire                    i_start,
    input  wire signed [NB_W-1:0]  i_eI,
    input  wire signed [NB_W-1:0]  i_eQ,

    // --- Salida: frame zero-padded (2N muestras) para FFT_ERROR ---
    output reg                     o_valid,
    output reg                     o_start,
    output reg  signed [NB_W-1:0]  o_eI,
    output reg  signed [NB_W-1:0]  o_eQ
);

    localparam integer N  = NFFT / 2;       // 16
    localparam integer KW = $clog2(N);      // 4 bits para 0..N-1

    // ============================================================
    // Buffer interno — RAM de N palabras complejas
    // ============================================================
    reg signed [NB_W-1:0] buf_I [0:N-1];
    reg signed [NB_W-1:0] buf_Q [0:N-1];

    // ============================================================
    // FSM
    // ============================================================
    localparam [1:0] ST_IDLE  = 2'd0,
                     ST_RECV  = 2'd1,
                     ST_ZEROS = 2'd2,
                     ST_ERROR = 2'd3;

    reg [1:0]  state;
    reg [KW:0] cnt;   // KW+1 bits para no hacer overflow en N-1

    always @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            cnt   <= {(KW+1){1'b0}};
        end else begin
            case (state)

                ST_IDLE: begin
                    if (i_valid && i_start) begin
                        buf_I[0] <= i_eI;
                        buf_Q[0] <= i_eQ;
                        cnt   <= {{KW{1'b0}}, 1'b1};
                        state <= ST_RECV;
                    end
                end

                ST_RECV: begin
                    if (i_valid) begin
                        buf_I[cnt[KW-1:0]] <= i_eI;
                        buf_Q[cnt[KW-1:0]] <= i_eQ;
                        if (cnt == N-1) begin
                            cnt   <= {(KW+1){1'b0}};
                            state <= ST_ZEROS;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                ST_ZEROS: begin
                    if (cnt == N-1) begin
                        cnt   <= {(KW+1){1'b0}};
                        state <= ST_ERROR;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                ST_ERROR: begin
                    if (cnt == N-1) begin
                        cnt   <= {(KW+1){1'b0}};
                        state <= ST_IDLE;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // ============================================================
    // Salida registrada — 1 ciclo de latencia sobre el estado
    // ============================================================
    always @(posedge clk) begin
        if (rst) begin
            o_valid <= 1'b0;
            o_start <= 1'b0;
            o_eI    <= {NB_W{1'b0}};
            o_eQ    <= {NB_W{1'b0}};
        end else begin
            case (state)

                ST_ZEROS: begin
                    o_valid <= 1'b1;
                    o_start <= (cnt == 0);
                    o_eI    <= {NB_W{1'b0}};
                    o_eQ    <= {NB_W{1'b0}};
                end

                ST_ERROR: begin
                    o_valid <= 1'b1;
                    o_start <= 1'b0;
                    o_eI    <= buf_I[cnt[KW-1:0]];
                    o_eQ    <= buf_Q[cnt[KW-1:0]];
                end

                default: begin
                    o_valid <= 1'b0;
                    o_start <= 1'b0;
                    o_eI    <= {NB_W{1'b0}};
                    o_eQ    <= {NB_W{1'b0}};
                end

            endcase
        end
    end

endmodule

`default_nettype wire
