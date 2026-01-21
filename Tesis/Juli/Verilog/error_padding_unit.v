`timescale 1ns / 1ps

module error_padding_unit #(
    parameter integer W = 16,     // Ancho de dato
    parameter integer N = 16      // Tamaño del bloque (mitad de la FFT)
)(
    input  wire clk,
    input  wire rst,

    // Entrada (Viene del Slicer)
    input  wire i_valid,
    input  wire i_last,
    input  wire signed [W-1:0] i_err_re,
    input  wire signed [W-1:0] i_err_im,

    // Salida (Va hacia la FFT de Error)
    output reg  o_valid,
    output reg  o_last,
    output reg  signed [W-1:0] o_data_re,
    output reg  signed [W-1:0] o_data_im
);

    // --- 1. Memoria (Sala de Espera) ---
    reg signed [W-1:0] mem_re [0:N-1];
    reg signed [W-1:0] mem_im [0:N-1];

    // --- 2. Punteros y Control ---
    // Función para calcular bits necesarios
    function integer clog2;
        input integer value;
        begin
            value = value - 1;
            for (clog2 = 0; value > 0; clog2 = clog2 + 1)
                value = value >> 1;
        end
    endfunction

    localparam P_BITS = clog2(N);
    
    reg [P_BITS:0] ptr_write;
    reg [P_BITS:0] ptr_read;
    reg state_flush; // 0: Recibiendo/Rellenando, 1: Vaciando memoria
    integer i;

    // --- 3. Lógica Principal ---
    always @(posedge clk) begin
        if (rst) begin
            o_valid     <= 0;
            o_last      <= 0;
            o_data_re   <= 0;
            o_data_im   <= 0;
            ptr_write   <= 0;
            ptr_read    <= 0;
            state_flush <= 0;
            // Limpieza de RAM (opcional para simulación)
            for(i=0; i<N; i=i+1) begin mem_re[i]=0; mem_im[i]=0; end
        end else begin
            
            // Valores por defecto (se sobrescriben si hay actividad)
            o_valid <= 0;
            o_last  <= 0;

            // --- ESTADO 0: FASE DE ENTRADA (PADDING) ---
            // Mientras llegan los datos, guardamos en RAM y sacamos CEROS.
            if (!state_flush) begin
                if (i_valid) begin
                    // A. Guardar en memoria
                    mem_re[ptr_write] <= i_err_re;
                    mem_im[ptr_write] <= i_err_im;

                    // B. Mandar Ceros afuera (Padding)
                    o_data_re <= 0;
                    o_data_im <= 0;
                    o_valid   <= 1;

                    // C. Actualizar Puntero de Escritura
                    if (ptr_write < N-1) begin
                        ptr_write <= ptr_write + 1;
                    end else begin
                        ptr_write <= 0; // Reiniciar por seguridad
                    end

                    // D. ¿Terminamos de recibir el bloque?
                    // Si llegó el último dato o llenamos el buffer, cambiamos de estado.
                    if (i_last || ptr_write == N-1) begin
                        state_flush <= 1; // Pasamos a vaciar
                        ptr_read    <= 0;
                    end
                end
            end 
            
            // --- ESTADO 1: FASE DE SALIDA (VACIADO) ---
            // Ya mandamos los ceros, ahora mandamos lo que guardamos en RAM.
            else begin
                // A. Leer de memoria y mandar afuera
                o_data_re <= mem_re[ptr_read];
                o_data_im <= mem_im[ptr_read];
                o_valid   <= 1;

                // B. Actualizar Puntero de Lectura
                if (ptr_read < N-1) begin
                    ptr_read <= ptr_read + 1;
                end else begin
                    // C. Fin del bloque completo (2N datos enviados)
                    o_last      <= 1;  // ¡Avisamos que terminó!
                    ptr_read    <= 0;
                    state_flush <= 0;  // Volvemos a esperar datos nuevos
                end
            end
        end
    end

endmodule