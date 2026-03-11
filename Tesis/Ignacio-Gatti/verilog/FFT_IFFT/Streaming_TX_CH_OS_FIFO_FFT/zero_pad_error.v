`timescale 1ns/1ps
`default_nettype none

// ============================================================
// zero_pad_error
//
// Propósito:
//   Primera etapa del loop LMS del PBFDAF.
//   Convierte el bloque de error e_blk (N muestras del slicer)
//   en un frame de 2N muestras listo para la FFT_ERROR:
//
//     Salida = [ 0, 0, ..., 0,  e[0], e[1], ..., e[N-1] ]
//               ←   N ceros  →  ←     N errores      →
//
//   Esto implementa el zero-padding causal del overlap-save:
//   los ceros representan el pasado (no hay error anterior),
//   y el error ocupa la segunda mitad del frame.
//
// Interfaz:
//   - Entrada:  stream de N muestras con i_valid/i_start
//               (conectado directamente a sl_out_e_* del slicer)
//   - Salida:   stream de 2N muestras con o_valid/o_start
//               listo para entrar a fft_ifft_stream (i_inverse=0)
//
// Implementación:
//   Contador eff_samp de 0 a 2N-1:
//     - eff_samp  0 ..  N-1  → o_eI=0, o_eQ=0  (zona de ceros)
//     - eff_samp  N .. 2N-1  → o_eI=i_eI, o_eQ=i_eQ  (zona de error)
//   La entrada solo avanza el contador cuando está en la zona de error
//   (eff_samp >= N), porque la entrada tiene N muestras, no 2N.
//   En la zona de ceros el contador avanza solo con el reloj.
//
// Timing:
//   - i_start arranca el frame: eff_samp se resetea a 0
//   - Latencia 0 ciclos en zona de ceros (combinacional)
//   - Latencia 1 ciclo en zona de error (registro de salida)
//   - o_start se emite en eff_samp=0 (primer cero del frame)
//
// Parámetros:
//   NB_W  = ancho de dato (9 bits, Q9.7)
//   NFFT  = 2N = tamaño del frame FFT (32)
// ============================================================

module zero_pad_error #(
    parameter integer NB_W = 9,
    parameter integer NFFT = 32    // 2N — tamaño del frame de salida
)(
    input  wire                    clk,
    input  wire                    rst,

    // --- Entrada: e_blk del slicer (N=NFFT/2 muestras) ---
    input  wire                    i_valid,   // válido 1 muestra cada ciclo
    input  wire                    i_start,   // primer error del bloque
    input  wire signed [NB_W-1:0]  i_eI,      // error Re
    input  wire signed [NB_W-1:0]  i_eQ,      // error Im

    // --- Salida: frame zero-padded (2N muestras) para FFT_ERROR ---
    output reg                     o_valid,
    output reg                     o_start,   // primer sample del frame (cero)
    output reg  signed [NB_W-1:0]  o_eI,      // 0 en primera mitad, e en segunda
    output reg  signed [NB_W-1:0]  o_eQ,

    // --- Debug ---
    output wire [$clog2(NFFT)-1:0] o_samp_idx // índice actual (0..2N-1)
);

    localparam integer N   = NFFT / 2;        // 16
    localparam integer KW  = $clog2(NFFT);    // 5
    localparam [KW-1:0] N_W = N[KW-1:0];      // 16 en KW bits

    // ============================================================
    // Contador eff_samp (0 .. 2N-1)
    // ============================================================
    reg [KW-1:0] eff_samp;

    // El contador avanza cuando:
    //   - Está en la zona de ceros (eff_samp < N): avanza con el reloj libre
    //     siempre que el módulo esté activo (armed=1)
    //   - Está en la zona de error (eff_samp >= N): avanza cuando i_valid=1
    //     (sincronizado con la entrada del slicer)
    reg armed;  // 1 tras recibir el primer i_start

    wire in_zero_zone  = (eff_samp <  N_W);
    wire in_error_zone = (eff_samp >= N_W);

    // Avance del contador
    wire cnt_advance = armed && (
        (in_zero_zone)              ||   // zona de ceros: avanza libre
        (in_error_zone && i_valid)       // zona de error: avanza con entrada
    );

    always @(posedge clk) begin
        if (rst) begin
            eff_samp <= {KW{1'b0}};
            armed    <= 1'b0;
        end else begin
            // Arrancar en el primer i_start
            if (i_start) begin
                eff_samp <= {KW{1'b0}};
                armed    <= 1'b1;
            end else if (cnt_advance) begin
                if (eff_samp == (NFFT-1))
                    eff_samp <= {KW{1'b0}};
                else
                    eff_samp <= eff_samp + 1'b1;
            end
        end
    end

    assign o_samp_idx = eff_samp;

    // ============================================================
    // Dato de salida combinacional
    // ============================================================
    wire samp_valid_comb = armed;               // válido desde el primer arranque
    wire samp_start_comb = (eff_samp == 0) && armed;  // primer sample del frame

    // En zona de ceros: salida = 0
    // En zona de error: salida = entrada del slicer
    wire signed [NB_W-1:0] eI_comb = in_error_zone ? i_eI : {NB_W{1'b0}};
    wire signed [NB_W-1:0] eQ_comb = in_error_zone ? i_eQ : {NB_W{1'b0}};

    // ============================================================
    // Registro de salida — 1 ciclo de latencia
    // ============================================================
    always @(posedge clk) begin
        if (rst) begin
            o_valid <= 1'b0;
            o_start <= 1'b0;
            o_eI    <= {NB_W{1'b0}};
            o_eQ    <= {NB_W{1'b0}};
        end else begin
            o_valid <= samp_valid_comb;
            o_start <= samp_start_comb;
            if (samp_valid_comb) begin
                o_eI <= eI_comb;
                o_eQ <= eQ_comb;
            end else begin
                o_eI <= {NB_W{1'b0}};
                o_eQ <= {NB_W{1'b0}};
            end
        end
    end

endmodule

`default_nettype wire
