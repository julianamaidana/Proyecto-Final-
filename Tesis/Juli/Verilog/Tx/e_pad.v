`timescale 1ns / 1ps

module error_padding_unit #(
    parameter integer W = 16,    // Ancho de dato
    parameter integer N = 16     // Tamaño de partición (Mitad de la FFT)
)(
    input  wire clk,
    input  wire rst,

    // -- Entrada desde el Slicer (Ráfaga de N muestras) --
    input  wire i_valid,
    input  wire i_last,
    input  wire signed [W-1:0] i_err_re,
    input  wire signed [W-1:0] i_err_im,

    // -- Salida hacia la FFT de Gradiente (Ráfaga de 2N muestras) --
    // Primero salen N ceros, luego salen N muestras de error.
    output reg  o_valid,
    output reg  o_last,
    output reg  signed [W-1:0] o_data_re,
    output reg  signed [W-1:0] o_data_im
);

    // Memoria para guardar el error mientras mandamos los ceros
    // Tamaño N (ej. 16)
    reg signed [W-1:0] mem_re [0:N-1];
    reg signed [W-1:0] mem_im [0:N-1];
    
    // Punteros y Contadores
    reg [$clog2(N):0] ptr_write;
    reg [$clog2(N):0] ptr_read;
    reg state_flush; // 0: Recibiendo/Zeros, 1: Vaciando/Datos

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            o_valid     <= 0;
            o_last      <= 0;
            o_data_re   <= 0;
            o_data_im   <= 0;
            ptr_write   <= 0;
            ptr_read    <= 0;
            state_flush <= 0;
            // Limpieza opcional de RAM para simulación limpia
            for(i=0; i<N; i=i+1) begin mem_re[i]=0; mem_im[i]=0; end
        end else begin
            
            // Lógica de Salida por defecto
            o_valid <= 0;
            o_last  <= 0;

            // --- ESTADO 0: Recibiendo Error / Generando Zeros ---
            if (!state_flush) begin
                if (i_valid) begin
                    // 1. Guardamos el dato real en la memoria
                    mem_re[ptr_write] <= i_err_re;
                    mem_im[ptr_write] <= i_err_im;
                    
                    // Avanzamos puntero de escritura
                    // (Nota: Si tu i_valid es continuo, esto cuenta de 0 a N-1)
                    if (ptr_write < N-1) begin
                        ptr_write <= ptr_write + 1;
                    end else begin
                        // Si llegamos al final (o i_last), preparamos cambio de estado
                        ptr_write <= 0; 
                    end

                    // 2. HACIA AFUERA: Mentimos y mandamos CEROS
                    o_data_re <= 0;
                    o_data_im <= 0;
                    o_valid   <= 1;

                    // Si terminó la entrada, pasamos a vaciar el buffer
                    if (i_last || ptr_write == N-1) begin
                        state_flush <= 1;
                        ptr_read    <= 0;
                    end
                end
            end 
            
            // --- ESTADO 1: Vaciando Buffer (Mandando el Error) ---
            else begin
                // Leemos de la memoria lo que guardamos antes
                o_data_re <= mem_re[ptr_read];
                o_data_im <= mem_im[ptr_read];
                o_valid   <= 1;

                if (ptr_read < N-1) begin
                    ptr_read <= ptr_read + 1;
                end else begin
                    // Terminamos de mandar los N datos guardados
                    o_last      <= 1; // Avisamos a la FFT que terminó el bloque de 2N
                    ptr_read    <= 0;
                    state_flush <= 0; // Volvemos a esperar nuevos datos
                end
            end
        end
    end

endmodule