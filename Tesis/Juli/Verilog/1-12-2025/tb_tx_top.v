`timescale 1ns / 1ps

module tb_tx_top;

    // ============================================================
    // 1. Declaración de Señales
    // ============================================================
    reg clk;
    reg rst_n;
    reg en;

    wire signed [15:0] tx_i;
    wire signed [15:0] tx_q;

    // Contador para saber por qué número vamos
    integer count;

    // ============================================================
    // 2. Instancia del DUT
    // ============================================================
    tx_top #(
        .DATA_WIDTH(16)
    ) u_dut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .tx_i  (tx_i),
        .tx_q  (tx_q)
    );

    // ============================================================
    // 3. Generación de Reloj
    // ============================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // ============================================================
    // 4. Estímulos
    // ============================================================
    initial begin
        $display("=== INICIO DE LA SIMULACION TX (1000 Símbolos) ===");
        rst_n = 0; 
        en    = 0; 
        #100;      

        // Soltar Reset
        @(negedge clk); 
        rst_n = 1; 
        #20;

        // Habilitar
        $display("[t=%0t] Habilitando TX...", $time);
        @(negedge clk);
        en = 1;

        // --- CAMBIO AQUÍ: Correr por 1000 símbolos ---
        // Esto cubre casi 2 veces el ciclo del PRBS9 (511 bits)
        repeat(1000) @(posedge clk);

        // Prueba de Pausa
        $display("[t=%0t] Pausando TX...", $time);
        en = 0;
        repeat(10) @(posedge clk);

        // Reanudar un poco más
        $display("[t=%0t] Reanudando...", $time);
        en = 1;
        repeat(50) @(posedge clk);

        $display("=== FIN DE LA SIMULACION ===");
        $finish;
    end

    // ============================================================
    // 5. Monitor Inteligente (Para no saturar la consola)
    // ============================================================
    initial count = 0;

    always @(posedge clk) begin
        if (rst_n && en) begin
            #1; // Esperar estabilidad
            count = count + 1;
            
            // Solo imprimimos cada 100 símbolos para ver que avanza
            // O si hay un cambio interesante
            if (count % 100 == 0) begin
                $display("Símbolo %0d | Time: %0t | I: %d | Q: %d", count, $time, tx_i, tx_q);
            end
        end
    end

endmodule