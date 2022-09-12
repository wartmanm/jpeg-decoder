use std::simd::*;
use std::simd::Which::{First, Second};
use std::ops::Shr;
use std::ops::Shl;
use std::ops::Mul;

// per godbolt this produces a pmulhw
// I'd be very surprised if there were a way to get a mulhrsw
#[inline(always)]
fn multiply_high(a: i16x8, b: i16x8) -> i16x8 {
    ((a.cast::<i32>() * b.cast::<i32>()) >> i32x8::splat(16)).cast::<i16>()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn idct8(data: &mut [i16x8; 8]) {
    // The fixed-point constants here are obtained by taking the fractional part of the constants
    // from the non-SIMD implementation and scaling them up by 1<<15. This is because
    // multiply_high(a, b) is effectively equivalent to (a*b)>>15 (except for possibly some
    // slight differences in rounding).

    // The code here is effectively equivalent to the calls to "kernel" in idct.rs, except that it
    // doesn't apply any further scaling and fixed point constants have a different precision.

    let p2 = data[2];
    let p3 = data[6];
    let p1 = multiply_high(p2.saturating_add(p3), i16x8::splat(17734)); // 0.5411961
    let t2 = i16x8::saturating_sub(
        i16x8::saturating_sub(p1, p3),
        multiply_high(p3, i16x8::splat(27779)), // 0.847759065
    );
    let t3 = i16x8::saturating_add(p1, multiply_high(p2, i16x8::splat(25079))); // 0.765366865

    println!("p2={:?}, p3={:?}, p1={:?}, t2={:?}, t3={:?}", p2, p3, p1, t2, t3);

    let p2 = data[0];
    let p3 = data[4];
    let t0 = i16x8::saturating_add(p2, p3);
    let t1 = i16x8::saturating_sub(p2, p3);

    println!("p2={:?}, p3={:?}, t0={:?}, t1={:?}", p2, p3, t0, t1);

    let x0 = i16x8::saturating_add(t0, t3);
    let x3 = i16x8::saturating_sub(t0, t3);
    let x1 = i16x8::saturating_add(t1, t2);
    let x2 = i16x8::saturating_sub(t1, t2);

    println!("x0={:?}, x3={:?}, x1={:?}, x2={:?}", x0, x3, x1, x2);

    let t0 = data[7];
    let t1 = data[5];
    let t2 = data[3];
    let t3 = data[1];

    let p3 = i16x8::saturating_add(t0, t2);
    let p4 = i16x8::saturating_add(t1, t3);
    let p1 = i16x8::saturating_add(t0, t3);
    let p2 = i16x8::saturating_add(t1, t2);
    let p5 = i16x8::saturating_add(p3, p4);
    let p5 = i16x8::saturating_add(p5, multiply_high(p5, i16x8::splat(5763))); // 0.175875602

    println!("p3={:?}, p4={:?}, p1={:?}, p2={:?}, p5={:?}", p3, p4, p1, p2, p5);

    let t0 = multiply_high(t0, i16x8::splat(9786)); // 0.298631336
    let t1 = i16x8::saturating_add(
        i16x8::saturating_add(t1, t1),
        multiply_high(t1, i16x8::splat(1741)), // 0.053119869
    );
    let t2 = i16x8::saturating_add(
        i16x8::saturating_add(t2, i16x8::saturating_add(t2, t2)),
        multiply_high(t2, i16x8::splat(2383)), // 0.072711026
    );
    let t3 = i16x8::saturating_add(t3, multiply_high(t3, i16x8::splat(16427))); // 0.501321110
    println!("t0={:?}, t1={:?}, t2={:?}, t3={:?}", t0, t1, t2, t3);

    let p1 = i16x8::saturating_sub(p5, multiply_high(p1, i16x8::splat(29490))); // 0.899976223
    let p2 = i16x8::saturating_sub(
        i16x8::saturating_sub(i16x8::saturating_sub(p5, p2), p2),
        multiply_high(p2, i16x8::splat(18446)), // 0.562915447
    );

    let p3 = i16x8::saturating_sub(
        multiply_high(p3, i16x8::splat(-31509)), // -0.961570560
        p3,
    );
    let p4 = multiply_high(p4, i16x8::splat(-12785)); // -0.390180644
    println!("p1={:?}, p2={:?}, p3={:?}, p4={:?}", p1, p2, p3, p4);

    let t3 = i16x8::saturating_add(i16x8::saturating_add(p1, p4), t3);
    let t2 = i16x8::saturating_add(i16x8::saturating_add(p2, p3), t2);
    let t1 = i16x8::saturating_add(i16x8::saturating_add(p2, p4), t1);
    let t0 = i16x8::saturating_add(i16x8::saturating_add(p1, p3), t0);
    println!("t0={:?}, t1={:?}, t2={:?}, t3={:?}", t0, t1, t2, t3);

    data[0] = i16x8::saturating_add(x0, t3);
    data[7] = i16x8::saturating_sub(x0, t3);
    data[1] = i16x8::saturating_add(x1, t2);
    data[6] = i16x8::saturating_sub(x1, t2);
    data[2] = i16x8::saturating_add(x2, t1);
    data[5] = i16x8::saturating_sub(x2, t1);
    data[3] = i16x8::saturating_add(x3, t0);
    data[4] = i16x8::saturating_sub(x3, t0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn transpose8(data: &mut [i16x8; 8]) {
    // Transpose a 8x8 matrix with a sequence of interleaving operations.
    // Naming: dABl contains elements from the *l*ower halves of vectors A and B, interleaved, i.e.
    // A0 B0 A1 B1 ...
    // dABCDll contains elements from the lower quarter (ll) of vectors A, B, C, D, interleaved -
    // A0 B0 C0 D0 A1 B1 C1 D1 ...
    //let d01l = simd_swizzle!(data[0], data[1], First(0), Second(0), First(1), Second(1), First(2), Second(2), First(3), Second(3));
    //let d23l = simd_swizzle!(data[2], data[3], First(0), Second(0), First(1), Second(1), First(2), Second(2), First(3), Second(3));
    //let d45l = simd_swizzle!(data[4], data[5], First(0), Second(0), First(1), Second(1), First(2), Second(2), First(3), Second(3));
    //let d67l = simd_swizzle!(data[6], data[7], First(0), Second(0), First(1), Second(1), First(2), Second(2), First(3), Second(3));
    //let d01h = simd_swizzle!(data[0], data[1], First(4), Second(4), First(5), Second(5), First(6), Second(6), First(7), Second(7));
    //let d23h = simd_swizzle!(data[2], data[3], First(4), Second(4), First(5), Second(5), First(6), Second(6), First(7), Second(7));
    //let d45h = simd_swizzle!(data[4], data[5], First(4), Second(4), First(5), Second(5), First(6), Second(6), First(7), Second(7));
    //let d67h = simd_swizzle!(data[6], data[7], First(4), Second(4), First(5), Second(5), First(6), Second(6), First(7), Second(7));

    let (d01l, d01h) = data[0].interleave(data[1]);
    let (d23l, d23h) = data[2].interleave(data[3]);
    let (d45l, d45h) = data[4].interleave(data[5]);
    let (d67l, d67h) = data[6].interleave(data[7]);

    //#[inline(always)]
    //fn interleave_pairs(a: i16x8, b: i16x8) -> (i16x8, i16x8) {
    //    let low = simd_swizzle!(a, b, [First(0), First(1), Second(0), Second(1), First(2), First(3), Second(2), Second(3)]);
    //    let high = simd_swizzle!(a, b, [First(4), First(5), Second(4), Second(5), First(6), First(7), Second(6), Second(7)]);
    //    (low, high)
    //}

    // Interleave consecutive pairs of 16-bit integers.
    // We are at the mercy of the optimizer's pattern recognition, but the best alternative seems
    // to be mem::transmute::<i16x8, i32x4>().
    let d0123ll = simd_swizzle!(d01l, d23l, [First(0), First(1), Second(0), Second(1), First(2), First(3), Second(2), Second(3)]);
    let d0123lh = simd_swizzle!(d01l, d23l, [First(4), First(5), Second(4), Second(5), First(6), First(7), Second(6), Second(7)]);
    let d4567ll = simd_swizzle!(d45l, d67l, [First(0), First(1), Second(0), Second(1), First(2), First(3), Second(2), Second(3)]);
    let d4567lh = simd_swizzle!(d45l, d67l, [First(4), First(5), Second(4), Second(5), First(6), First(7), Second(6), Second(7)]);
    let d0123hl = simd_swizzle!(d01h, d23h, [First(0), First(1), Second(0), Second(1), First(2), First(3), Second(2), Second(3)]);
    let d0123hh = simd_swizzle!(d01h, d23h, [First(4), First(5), Second(4), Second(5), First(6), First(7), Second(6), Second(7)]);
    let d4567hl = simd_swizzle!(d45h, d67h, [First(0), First(1), Second(0), Second(1), First(2), First(3), Second(2), Second(3)]);
    let d4567hh = simd_swizzle!(d45h, d67h, [First(4), First(5), Second(4), Second(5), First(6), First(7), Second(6), Second(7)]);

    //#[inline(always)]
    //fn interleave_quad(a: i16x8, b: i16x8) -> (i16x8, i16x8) {
    //    let low = simd_swizzle!(a, b, [First(0), First(1), First(2), First(3), Second(0), Second(1), Second(2), Second(3)]);
    //    let high = simd_swizzle!(a, b, [First(4), First(5), First(6), First(7), Second(4), Second(5), Second(6), Second(7)]);
    //    (low, high)
    //}
    // Interleave consecutive quadruples of 16-bit integers.
    data[0] = simd_swizzle!(d0123ll, d4567ll, [First(0), First(1), First(2), First(3), Second(0), Second(1), Second(2), Second(3)]);
    data[1] = simd_swizzle!(d0123ll, d4567ll, [First(4), First(5), First(6), First(7), Second(4), Second(5), Second(6), Second(7)]);
    data[2] = simd_swizzle!(d0123lh, d4567lh, [First(0), First(1), First(2), First(3), Second(0), Second(1), Second(2), Second(3)]);
    data[3] = simd_swizzle!(d0123lh, d4567lh, [First(4), First(5), First(6), First(7), Second(4), Second(5), Second(6), Second(7)]);
    data[4] = simd_swizzle!(d0123hl, d4567hl, [First(0), First(1), First(2), First(3), Second(0), Second(1), Second(2), Second(3)]);
    data[5] = simd_swizzle!(d0123hl, d4567hl, [First(4), First(5), First(6), First(7), Second(4), Second(5), Second(6), Second(7)]);
    data[6] = simd_swizzle!(d0123hh, d4567hh, [First(0), First(1), First(2), First(3), Second(0), Second(1), Second(2), Second(3)]);
    data[7] = simd_swizzle!(d0123hh, d4567hh, [First(4), First(5), First(6), First(7), Second(4), Second(5), Second(6), Second(7)]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
pub unsafe fn dequantize_and_idct_block_8x8(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
) {
    // The loop below will write to positions [output_linestride * i, output_linestride * i + 8)
    // for 0<=i<8. Thus, the last accessed position is at an offset of output_linestrade * 7 + 7,
    // and if that position is in-bounds, so are all other accesses.
    assert!(
        output.len()
            > output_linestride
                .checked_mul(7)
                .unwrap()
                .checked_add(7)
                .unwrap()
    );

    const SHIFT: i16 = 3;

    // Read the DCT coefficients, scale them up and dequantize them.
    let mut data = [i16x8::splat(0); 8];
    for i in 0..8 {
        data[i] = i16x8::shl(
            i16x8::mul(
                std::ptr::read(coefficients.as_ptr().wrapping_add(i * 8) as *const _),
                // TODO
                std::ptr::read(quantization_table.as_ptr().wrapping_add(i * 8) as *const i16x8),
            ),
            i16x8::splat(SHIFT)
        );
    }

    // Usual column IDCT - transpose - column IDCT - transpose approach.
    idct8(&mut data);
    transpose8(&mut data);
    idct8(&mut data);
    transpose8(&mut data);

    for i in 0..8 {
        let mut buf = [0u8; 16];
        // The two passes of the IDCT algorithm give us a factor of 8, so the shift here is
        // increased by 3.
        // As values will be stored in a u8, they need to be 128-centered and not 0-centered.
        // We add 128 with the appropriate shift for that purpose.
        const OFFSET: i16 = 128 << (SHIFT + 3);
        // We want rounding right shift, so we should add (1/2) << (SHIFT+3) before shifting.
        const ROUNDING_BIAS: i16 = (1 << (SHIFT + 3)) >> 1;

        let data_with_offset = i16x8::saturating_add(data[i], i16x8::splat(OFFSET + ROUNDING_BIAS));

        std::ptr::write::<u8x8>(
            buf.as_mut_ptr() as *mut _,
            i16x8::shr(data_with_offset, i16x8::splat(SHIFT + 3)).cast::<u8>(),
        );
        std::ptr::copy_nonoverlapping::<u8>(
            buf.as_ptr(),
            output.as_mut_ptr().wrapping_add(output_linestride * i) as *mut _,
            8,
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
pub unsafe fn color_convert_line_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len() % 3 == 0);
    let num = output.len() / 3;
    assert!(num <= y.len());
    assert!(num <= cb.len());
    assert!(num <= cr.len());
    // _mm_loadu_si64 generates incorrect code for Rust <1.58. To circumvent this, we use a full
    // 128-bit load, but that requires leaving an extra vector of border to the scalar code.
    // From Rust 1.58 on, the _mm_loadu_si128 can be replaced with _mm_loadu_si64 and this
    // .saturating_sub() can be removed.
    let num_vecs = (num / 8).saturating_sub(1);

    for i in 0..num_vecs {
        const SHIFT: i16 = 6;
        // Load.
        let y: i8x8 = std::ptr::read(y.as_ptr().wrapping_add(i * 8) as *const _);
        let cb: i8x8 = std::ptr::read(cb.as_ptr().wrapping_add(i * 8) as *const _);
        let cr: i8x8 = std::ptr::read(cr.as_ptr().wrapping_add(i * 8) as *const _);

        // Convert to 16 bit.
        let y = y.cast::<i16>();
        let cb = cb.cast::<i16>();
        let cr = cr.cast::<i16>();

        // Add offsets
        let c128 = i16x8::splat(128 << SHIFT);
        // TODO
        let y = i16x8::saturating_add(y, i16x8::splat((1 << SHIFT) >> 1));
        let cb = i16x8::saturating_sub(cb, c128);
        let cr = i16x8::saturating_sub(cr, c128);

        // Compute cr * 1.402, cb * 0.34414, cr * 0.71414, cb * 1.772
        let cr_140200 = i16x8::saturating_add(multiply_high(cr, i16x8::splat(13173)), cr);
        let cb_034414 = multiply_high(cb, i16x8::splat(11276));
        let cr_071414 = multiply_high(cr, i16x8::splat(23401));
        let cb_177200 = i16x8::saturating_add(multiply_high(cb, i16x8::splat(25297)), cb);

        // Last conversion step.
        let r = i16x8::saturating_add(y, cr_140200);
        let g = i16x8::saturating_sub(y, i16x8::saturating_add(cb_034414, cr_071414));
        let b = i16x8::saturating_add(y, cb_177200);

        // Shift back and convert to u8.
        let r = i16x8::shr(r, i16x8::splat(SHIFT)).cast::<u8>();
        let g = i16x8::shr(g, i16x8::splat(SHIFT)).cast::<u8>();
        let b = i16x8::shr(b, i16x8::splat(SHIFT)).cast::<u8>();

        // Shuffle rrrrrrrrggggggggbbbbbbbb to rgbrgbrgb...

        let rg_lanes = simd_swizzle!(r, g, [First(0), Second(0), First(0),
                                            First(1), Second(1), First(0),
                                            First(2), Second(2), First(0),
                                            First(3), Second(3), First(0),
                                            First(4), Second(4), First(0),
                                            First(5)]);
        let b_lanes = simd_swizzle!(b, [0, 0, 0,
                                        0, 0, 1,
                                        0, 0, 2,
                                        0, 0, 3,
                                        0, 0, 4,
                                        0]);
        let rb_xor = simd_swizzle!(r, b, [Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0)]);
        let rgb_low: u8x16 = rg_lanes ^ b_lanes ^ rb_xor;

        //let gb_lanes = simd_swizzle!(g, b, [First(5), Second(5), First(0),
        //                                    First(6), Second(6), First(0),
        //                                    First(7), Second(7), First(0),
        //                                    First(8), Second(8), First(0),
        //                                    First(9), Second(9), First(0),
        //                                    First(10)]);
        //let r_lanes = simd_swizzle!(r, r, [First(0), First(0), First(6),
        //                                   First(0), First(0), First(7),
        //                                   First(0), First(0), First(8),
        //                                   First(0), First(0), First(9),
        //                                   First(0), First(0), First(10),
        //                                   First(0)]);
        let gb_lanes = simd_swizzle!(g, b, [First(5), Second(5), First(0),
                                            First(6), Second(6), First(0),
                                            First(7), Second(7), First(0),
                                            First(0), Second(0), First(0),
                                            First(0), Second(0), First(0),
                                            First(0)]);
        let r_lanes = simd_swizzle!(r, r, [First(0), First(0), First(6),
                                           First(0), First(0), First(7),
                                           First(0), First(0), First(0),
                                           First(0), First(0), First(0),
                                           First(0), First(0), First(0),
                                           First(0)]);
        let gr_xor = simd_swizzle!(g, r, [Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0), Second(0), First(0),
                                          Second(0)]);
        let rgb_hi: u8x16 = gb_lanes ^ r_lanes ^ gr_xor;

        let mut data = [0u8; 32];
        std::ptr::write::<u8x16>(data.as_mut_ptr() as *mut _, rgb_low);
        std::ptr::write::<u8x16>(data.as_mut_ptr().wrapping_add(16) as *mut _, rgb_hi);
        std::ptr::copy_nonoverlapping::<u8>(
            data.as_ptr(),
            output.as_mut_ptr().wrapping_add(24 * i),
            24,
        );
    }

    num_vecs * 8
}
