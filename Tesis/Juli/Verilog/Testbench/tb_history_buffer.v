`timescale 1ns / 1ps

module tb_history_buffer;

    // Señales del Testbench
    reg clk = 0;
    reg rst = 1;
    reg i_valid = 0;
    reg signed [15:0] i_X_re, i_X_im;
    
    // COEFICIENTES DE PRUEBA (Formato Q2.14)
    reg signed [15:0] i_W0_re = 16'd16384; // W0 = 1.0
    reg signed [15:0] i_W0_im = 0;
    
    reg signed [15:0] i_W1_re = 16'd16384; // W1 = 1.0
    reg signed [15:0] i_W1_im = 0;

    // Salidas del módulo
    wire o_valid;
    wire signed [15:0] o_Y_re, o_Y_im;
    wire [4:0] o_k;

    // Instancia del Módulo a Probar 
    history_buffer #(
        .W(16), 
        .FRAC(14)
    ) uut (
        .clk(clk), .rst(rst),
        .i_valid(i_valid),
        .i_X_re(i_X_re), .i_X_im(i_X_im),
        .i_W0_re(i_W0_re), .i_W0_im(i_W0_im),
        .i_W1_re(i_W1_re), .i_W1_im(i_W1_im),
        .o_valid(o_valid),
        .o_Y_re(o_Y_re), .o_Y_im(o_Y_im),
        .o_k_idx(o_k)
    );

    // Generador de Reloj (10ns periodo = 100MHz)
    always #5 clk = ~clk;

    integer i;

    // --- PROCESO DE ESTÍMULOS ---
    initial begin
        // Secuencia de Reset
        rst = 1; #20;
        rst = 0; #10;

        // ------------------------------------------------
        // BLOQUE 1: Enviamos valor '1000'
        // Historial interno: Vacio (0)
        // Calculo esperado: (1000 * W0) + (0 * W1) = 1000 * 1 = 1000
        // ------------------------------------------------
        $display("\n--- INICIANDO BLOQUE 1 (Carga Inicial) ---");
        for (i=0; i<32; i=i+1) begin
            @(posedge clk);
            i_valid <= 1;
            i_X_re <= 16'd1000; 
            i_X_im <= 0;
        end
        
        @(posedge clk) i_valid <= 0; // Pausa
        #100;

        // ------------------------------------------------
        // BLOQUE 2: Enviamos valor '2000'
        // Historial interno: Tiene '1000' del bloque anterior
        // Calculo esperado: (2000 * W0) + (1000 * W1)
        //                 = (2000 * 1)  + (1000 * 1) = 3000
        // ------------------------------------------------
        $display("\n--- INICIANDO BLOQUE 2 (Prueba de Suma Histórica) ---");
        for (i=0; i<32; i=i+1) begin
            @(posedge clk);
            i_valid <= 1;
            i_X_re <= 16'd2000; 
            i_X_im <= 0;
        end
        
        @(posedge clk) i_valid <= 0;
        #100;
        $display("\n--- FIN DE SIMULACION ---");
        $finish;
    end

    // --- MONITOR DE SALIDA (Tu "Ojo" en la Terminal) ---
    always @(posedge clk) begin
        if (o_valid) begin
            $display("[T=%0t] Salida: Y_re=%d", $time, o_Y_re);

            // Verificación Automática
            // Caso Bloque 1 (Solo W0)
            if (o_Y_re == 1000) 
                $display("    -> [OK] Bloque 1 Correcto");
                
            // Caso Bloque 2 (W0 + W1 del pasado)
            else if (o_Y_re == 3000) 
                $display("    -> [OK] Bloque 2 Correcto (2000 + 1000)");
                
            else 
                $display("    -> [ERROR] Valor inesperado. Revisa la logica.");
        end
    end

endmodule