`timescale 1ns / 1ps

module tb_padding;

    // --- Configuración ---
    parameter W = 16;
    parameter N = 4; // Usamos 4 para ver fácil la secuencia (4 ceros -> 4 datos)

    // --- Señales ---
    reg clk = 0;
    reg rst = 0;
    reg i_valid = 0;
    reg i_last = 0;
    reg signed [W-1:0] i_err_re = 0;
    reg signed [W-1:0] i_err_im = 0;

    wire o_valid;
    wire o_last;
    wire signed [W-1:0] o_data_re;
    wire signed [W-1:0] o_data_im;

    // --- Generación de Reloj (10ns) ---
    always #5 clk = ~clk;

    // --- Instancia del Módulo (DUT) ---
    error_padding_unit #(
        .W(W),
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .i_valid(i_valid),
        .i_last(i_last),
        .i_err_re(i_err_re),
        .i_err_im(i_err_im),
        .o_valid(o_valid),
        .o_last(o_last),
        .o_data_re(o_data_re),
        .o_data_im(o_data_im)
    );

    // --- Proceso de Prueba ---
    initial begin
        // 1. Inicialización
        rst = 1;
        #20;
        rst = 0;
        #20;

        $display("--- Inicio de Test N=%0d ---", N);

        // 2. Enviar Paquete 1 (Valores 10, 11, 12, 13)
        enviar_paquete(10); 
        
        // Esperar un poco para ver el hueco entre paquetes
        #40; 

        // 3. Enviar Paquete 2 (Valores 20, 21, 22, 23)
        // Esto verifica que los punteros se hayan reiniciado bien
        enviar_paquete(20);

        #100;
        $display("--- Fin de Test ---");
        $finish;
    end

    // --- Tarea Auxiliar para enviar N datos ---
    task enviar_paquete(input integer start_val);
        integer k;
        begin
            $display("Enviando ráfaga iniciando en %0d...", start_val);
            wait(clk == 0); // Sincronizar
            
            for (k = 0; k < N; k = k + 1) begin
                @(posedge clk); // Esperar flanco
                i_valid  <= 1;
                i_err_re <= start_val + k; // Rampa: 10, 11, 12...
                i_err_im <= start_val + k; 
                
                // Levantar flag 'last' en el último dato
                if (k == N-1) i_last <= 1;  
                else          i_last <= 0;
            end
            
            // Apagar valid después del último
            @(posedge clk);
            i_valid <= 0;
            i_last  <= 0;
            i_err_re <= 0;
            
            // Esperar a que el módulo termine de vaciar (N ciclos de entrada + N de salida)
            // Damos un margen
            repeat (2*N + 4) @(posedge clk);
        end
    endtask

endmodule