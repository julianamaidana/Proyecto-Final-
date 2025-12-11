`timescale 1ns / 1ps

module tb_tx_vector_match;

    // --- PARÁMETROS ---
    parameter DATA_WIDTH = 16;
    parameter N_VECTORS  = 1000; // Debe coincidir con Python
    
    // Nombres de archivos (deben estar en la carpeta de simulación)
    parameter FILE_I = "tx_ref_i.hex";
    parameter FILE_Q = "tx_ref_q.hex";

    // --- SEÑALES ---
    reg clk = 0;
    reg rst_n = 0;
    reg en = 0;
    wire signed [DATA_WIDTH-1:0] tx_i;
    wire signed [DATA_WIDTH-1:0] tx_q;

    // --- MEMORIAS PARA GUARDAR REFERENCIA ---
    reg [DATA_WIDTH-1:0] ref_mem_i [0:N_VECTORS-1];
    reg [DATA_WIDTH-1:0] ref_mem_q [0:N_VECTORS-1];

    integer i;
    integer errors = 0;

    // --- DUT (Device Under Test) ---
    tx_top #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_dut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .tx_i  (tx_i),
        .tx_q  (tx_q)
    );

    // --- RELOJ (100 MHz) ---
    always #5 clk = ~clk; 

    // --- PROCESO DE PRUEBA ---
    initial begin
        // 1. Cargar vectores de referencia
        $readmemh(FILE_I, ref_mem_i);
        $readmemh(FILE_Q, ref_mem_q);
        
        $display("Iniciando Vector Matching TX...");

        // 2. Reset
        rst_n = 0;
        en    = 0;
        #100;
        
        rst_n = 1;
        // Sincronizar Enable con flanco de bajada para que arranque limpio
        @(negedge clk);
        en = 1;

        // 3. Comparación Ciclo a Ciclo
        for (i = 0; i < N_VECTORS; i = i + 1) begin
            // Esperar al flanco de subida donde el dato es válido
            @(posedge clk); 
            #1; // Pequeño retardo para leer el dato estable

            // Verificar Rama I
            if (tx_i !== ref_mem_i[i]) begin
                $display("[ERROR I] Idx %0d: Verilog=%h vs Python=%h", i, tx_i, ref_mem_i[i]);
                errors = errors + 1;
            end

            // Verificar Rama Q
            if (tx_q !== ref_mem_q[i]) begin
                $display("[ERROR Q] Idx %0d: Verilog=%h vs Python=%h", i, tx_q, ref_mem_q[i]);
                errors = errors + 1;
            end
        end

        // 4. Reporte Final
        $display("--------------------------------");
        if (errors == 0) begin
            $display("    ¡EXITO! VECTOR MATCHING PASADO.");
            $display("    El hardware TX es idéntico al Python.");
        end else begin
            $display("    FALLO: Se encontraron %0d errores.", errors);
        end
        $display("--------------------------------");
        
        $stop; // Detener simulación
    end

endmodule