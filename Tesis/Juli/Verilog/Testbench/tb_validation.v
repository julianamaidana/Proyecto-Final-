module tb_validation;
    reg clk = 0;
    reg rst = 1;
    reg signed [10:0] sigma_scale = 11'sd0;
    reg tb_tx_en = 0;
    reg i_valid_from_tb = 0; // Declarada como reg para controlarla
    wire buf_ready_out;

    // Conexiones de salida
    wire o_clean_valid, o_check_valid;
    wire [8:0] o_clean_I, o_clean_Q, o_check_I, o_check_Q;

    always #5 clk = ~clk;

    top_validation u_dut (
        .clk(clk), .rst(rst), .sigma_scale(sigma_scale),
        .tb_tx_en(tb_tx_en), 
        .i_valid_from_tb(i_valid_from_tb),
        .o_check_valid(o_check_valid), .o_check_I(o_check_I), .o_check_Q(o_check_Q),
        .o_clean_valid(o_clean_valid), .o_clean_I(o_clean_I), .o_clean_Q(o_clean_Q),
        .buf_ready_out(buf_ready_out)
    );

    initial begin
        tb_tx_en = 0; i_valid_from_tb = 0; rst = 1;
        #100 rst = 0;
        
        @(posedge clk);
        if (buf_ready_out) tb_tx_en = 1; // Arranca PRBS

        repeat (3) @(posedge clk); // Espera latencia del canal
        i_valid_from_tb = 1;      // Sincroniza el buffer
        
        // Loop de mantenimiento
        repeat (5000) begin
            @(posedge clk);
            if (!buf_ready_out) begin
                tb_tx_en <= 0; i_valid_from_tb <= 0;
            end else begin
                tb_tx_en <= 1; i_valid_from_tb <= 1;
            end
        end
        $stop;
    end
endmodule