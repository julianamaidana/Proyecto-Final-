module fft_ifft #(
    parameter integer NFFT        = 32,
    parameter integer LOGN        = 5,   // log2(NFFT)

    parameter integer NB_IN       = 9,
    parameter integer NBF_IN      = 7,

    parameter integer NB_W        = 17,
    parameter integer NBF_W       = 10,

    parameter integer NB_OUT      = 9,
    parameter integer NBF_OUT     = 7,

    parameter integer SCALE_STAGE = 0
)(
    input  wire                     i_clk,
    input  wire                     i_rst,

    input  wire                     i_valid,
    input  wire signed [NB_IN-1:0]  i_xI,
    input  wire signed [NB_IN-1:0]  i_xQ,
    input  wire                     i_inverse,  // 0=FFT, 1=IFFT

    output wire                     o_in_ready,

    output reg                      o_start,
    output reg                      o_valid,
    output reg  signed [NB_OUT-1:0] o_yI,
    output reg  signed [NB_OUT-1:0] o_yQ
);

    // ============================================================
    // FSM
    // ============================================================
    localparam [1:0] S_COLLECT = 2'd0;
    localparam [1:0] S_COMP    = 2'd1;
    localparam [1:0] S_SEND    = 2'd2;

    reg [1:0] state;
    assign o_in_ready = (state == S_COLLECT);

    // ============================================================
    // Buffers in-place (wide)
    // ============================================================
    reg signed [NB_W-1:0] bufI [0:NFFT-1];
    reg signed [NB_W-1:0] bufQ [0:NFFT-1];
    integer j;

    // ============================================================
    // Bit-reversal for DIT (load in bitrev order -> output natural order)
    // ============================================================
    function [LOGN-1:0] bitrev;
        input [LOGN-1:0] x;
        integer b;
        begin
            for (b = 0; b < LOGN; b = b + 1)
                bitrev[b] = x[LOGN-1-b];
        end
    endfunction

    // ============================================================
    // Input narrow -> wide (keep numeric value)
    // wide = signext(narrow) << (NBF_W - NBF_IN)
    // ============================================================
    wire signed [NB_W-1:0] inI_w = ($signed({{(NB_W-NB_IN){i_xI[NB_IN-1]}}, i_xI})) <<< (NBF_W - NBF_IN);
    wire signed [NB_W-1:0] inQ_w = ($signed({{(NB_W-NB_IN){i_xQ[NB_IN-1]}}, i_xQ})) <<< (NBF_W - NBF_IN);

    // ============================================================
    // Counters
    // ============================================================
    reg [LOGN-1:0] in_cnt;
    reg [LOGN-1:0] send_idx;

    reg [LOGN-1:0] stage;   // 0..LOGN-1
    reg [LOGN-1:0] k_idx;   // 0..half-1
    reg [LOGN-1:0] base;    // 0..NFFT-m step m

    // stage derived
    wire [LOGN:0] m    = (1 << (stage + 1));
    wire [LOGN:0] half = (1 << stage);
    wire [LOGN:0] step = (NFFT >> (stage + 1)); // NFFT/m

    wire [LOGN-1:0] i0 = base + k_idx;
    wire [LOGN-1:0] i1 = base + k_idx + half[LOGN-1:0];

    // For NFFT=32: tw_idx in 0..15
    wire [4:0] tw_idx = (k_idx * step[LOGN-1:0]);

    // ============================================================
    // Twiddle (FFT sign in ROM), for IFFT flip imag sign
    // ============================================================
    wire signed [NB_W-1:0] w_re_rom;
    wire signed [NB_W-1:0] w_im_fft_rom;

    twiddle_rom #(.NB_W(NB_W), .NBF_W(NBF_W)) u_tw (
        .i_k(tw_idx),
        .o_re(w_re_rom),
        .o_im_fft(w_im_fft_rom)
    );

    wire signed [NB_W-1:0] wI = w_re_rom;
    wire signed [NB_W-1:0] wQ = (i_inverse) ? -w_im_fft_rom : w_im_fft_rom;

    // ============================================================
    // Butterfly combinational for current (i0,i1)
    // ============================================================
    wire signed [NB_W-1:0] aI, aQ, bI, bQ;

    butterfly #(.NB_W(NB_W), .NBF_W(NBF_W), .SCALE(SCALE_STAGE)) u_bf (
        .i_uI(bufI[i0]), .i_uQ(bufQ[i0]),
        .i_vI(bufI[i1]), .i_vQ(bufQ[i1]),
        .i_wI(wI),       .i_wQ(wQ),
        .o_aI(aI), .o_aQ(aQ),
        .o_bI(bI), .o_bQ(bQ)
    );

    // ============================================================
    // IFFT final scaling: 1/NFFT (NFFT power-of-2) => >> LOGN with round-even
    // (match Python: invN al final)
    // ============================================================
    function signed [NB_W-1:0] rshift_round_even;
        input signed [NB_W-1:0] x;
        input integer sh;
        reg signed [NB_W-1:0] y;
        reg guard;
        reg sticky;
        integer t;
        begin
            if (sh <= 0) begin
                rshift_round_even = x;
            end else begin
                y     = x >>> sh;
                guard = x[sh-1];

                sticky = 1'b0;
                // sticky = OR de los bits descartados por debajo del guard (sh-2 .. 0)
                for (t = 0; t < sh-1; t = t + 1)
                    sticky = sticky | x[t];

                // ties-to-even: inc = guard & (sticky | LSB_kept)
                rshift_round_even = y + (guard & (sticky | y[0]));
            end
        end
    endfunction

    wire signed [NB_W-1:0] outI_w = (i_inverse) ? rshift_round_even(bufI[send_idx], LOGN) : bufI[send_idx];
    wire signed [NB_W-1:0] outQ_w = (i_inverse) ? rshift_round_even(bufQ[send_idx], LOGN) : bufQ[send_idx];

    // ============================================================
    // Output wide -> narrow
    // ============================================================
    wire signed [NB_OUT-1:0] yI_n;
    wire signed [NB_OUT-1:0] yQ_n;

    sat_trunc #(
        .NB_XI(NB_W), .NBF_XI(NBF_W),
        .NB_XO(NB_OUT), .NBF_XO(NBF_OUT),
        .ROUND_EVEN(1)
    ) u_outI (
        .i_data(outI_w),
        .o_data(yI_n)
    );

    sat_trunc #(
        .NB_XI(NB_W), .NBF_XI(NBF_W),
        .NB_XO(NB_OUT), .NBF_XO(NBF_OUT),
        .ROUND_EVEN(1)
    ) u_outQ (
        .i_data(outQ_w),
        .o_data(yQ_n)
    );

    // ============================================================
    // Sequential control
    // ============================================================
    always @(posedge i_clk) begin
        if (i_rst) begin
            state    <= S_COLLECT;
            in_cnt   <= 0;
            send_idx <= 0;
            stage    <= 0;
            k_idx    <= 0;
            base     <= 0;

            o_start  <= 1'b0;
            o_valid  <= 1'b0;
            o_yI     <= 0;
            o_yQ     <= 0;

            for (j = 0; j < NFFT; j = j + 1) begin
                bufI[j] <= 0;
                bufQ[j] <= 0;
            end
        end else begin
            o_start <= 1'b0;

            case (state)

                // --------------------------
                // COLLECT: load NFFT samples
                // --------------------------
                S_COLLECT: begin
                    o_valid <= 1'b0;
                    if (i_valid) begin
                        bufI[bitrev(in_cnt)] <= inI_w;
                        bufQ[bitrev(in_cnt)] <= inQ_w;

                        if (in_cnt == NFFT-1) begin
                            in_cnt <= 0;
                            stage  <= 0;
                            k_idx  <= 0;
                            base   <= 0;
                            state  <= S_COMP;
                        end else begin
                            in_cnt <= in_cnt + 1'b1;
                        end
                    end
                end

                // --------------------------
                // COMP: 1 butterfly per cycle
                // --------------------------
                S_COMP: begin
                    // write-back
                    bufI[i0] <= aI;  bufQ[i0] <= aQ;
                    bufI[i1] <= bI;  bufQ[i1] <= bQ;

                    // next indices
                    if (k_idx == half-1) begin
                        k_idx <= 0;

                        if (base + m >= NFFT) begin
                            base <= 0;

                            if (stage == LOGN-1) begin
                                send_idx <= 0;
                                o_start  <= 1'b1;
                                state    <= S_SEND;
                            end else begin
                                stage <= stage + 1'b1;
                            end
                        end else begin
                            base <= base + m[LOGN-1:0];
                        end

                    end else begin
                        k_idx <= k_idx + 1'b1;
                    end
                end

                // --------------------------
                // SEND: output NFFT samples
                // --------------------------
                S_SEND: begin
                    o_valid <= 1'b1;
                    o_yI    <= yI_n;
                    o_yQ    <= yQ_n;

                    if (send_idx == NFFT-1) begin
                        send_idx <= 0;
                        state    <= S_COLLECT;
                    end else begin
                        send_idx <= send_idx + 1'b1;
                    end
                end

                default: state <= S_COLLECT;

            endcase
        end
    end

endmodule
