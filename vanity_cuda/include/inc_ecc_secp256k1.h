/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

#ifndef INC_ECC_SECP256K1_H
#define INC_ECC_SECP256K1_H


// Generator point G coordinates for constant memory (reordered from G0-G7)
#define SECP256K1_GX0 0x16f81798
#define SECP256K1_GX1 0x59f2815b
#define SECP256K1_GX2 0x2dce28d9
#define SECP256K1_GX3 0x029bfcdb
#define SECP256K1_GX4 0xce870b07
#define SECP256K1_GX5 0x55a06295
#define SECP256K1_GX6 0xf9dcbbac
#define SECP256K1_GX7 0x79be667e

#define SECP256K1_GY0 0xfb10d4b8
#define SECP256K1_GY1 0x9c47d08f
#define SECP256K1_GY2 0xa6855419
#define SECP256K1_GY3 0xfd17b448
#define SECP256K1_GY4 0x0e1108a8
#define SECP256K1_GY5 0x5da4fbfc
#define SECP256K1_GY6 0x26a3c465
#define SECP256K1_GY7 0x483ada77

// the base point G in compressed form for parse_public
// parity and reversed byte/char (8 bit) byte order
// G = 02 79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798
#define SECP256K1_G_STRING0 0x66be7902
#define SECP256K1_G_STRING1 0xbbdcf97e
#define SECP256K1_G_STRING2 0x62a055ac
#define SECP256K1_G_STRING3 0x0b87ce95
#define SECP256K1_G_STRING4 0xfc9b0207
#define SECP256K1_G_STRING5 0x28ce2ddb
#define SECP256K1_G_STRING6 0x81f259d9
#define SECP256K1_G_STRING7 0x17f8165b
#define SECP256K1_G_STRING8 0x00000098

// pre computed values, can be verified using private keys for
// x1 is the same as the basepoint g
// x1 WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn
// x3 WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU74sHUHy8S
// x5 WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU75s2EPgZf
// x7 WIF: KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU76rnZwVdz


// 8+1 to make room for the parity

// (32*8 == 256)

#endif // INC_ECC_SECP256K1_H