/*
** ###################################################################
**     Processors:          MIMXRT1176AVM8A_cm7
**                          MIMXRT1176CVM8A_cm7
**                          MIMXRT1176DVMAA_cm7
**
**     Compiler:            GNU C Compiler
**     Reference manual:    IMXRT1170RM, Rev E, 12/2019
**     Version:             rev. 0.1, 2018-03-05
**     Build:               b200828
**
**     Abstract:
**         Linker file for the GNU C Compiler
**
**     Copyright 2016 Freescale Semiconductor, Inc.
**     Copyright 2016-2020 NXP
**     All rights reserved.
**
**     SPDX-License-Identifier: BSD-3-Clause
**
**     http:                 www.nxp.com
**     mail:                 support@nxp.com
**
** ###################################################################
*/

/* Entry Point */
ENTRY(Reset_Handler)

HEAP_SIZE  = DEFINED(__heap_size__)  ? __heap_size__  : 0x03F09800;
STACK_SIZE = DEFINED(__stack_size__) ? __stack_size__ : 0x0400;
RPMSG_SHMEM_SIZE = DEFINED(__use_shmem__) ? 0x2000 : 0;
NCACHE_SIZE = 0x8000;
TEXT_SIZE = 0x5B400;

/*
 * Valid memory regions:
 * 0x00000000 - 0x0003FFFF (256KB)
 * 0x20000000 - 0x2003FFFF (256KB)
 * 0x20240000 - 0x202BFFFF (512KB)
 * 0x202C0000 - 0x2033FFFF (512KB)
 */

/* Specify the memory areas */
MEMORY
{
  m_interrupts          (RX)  : ORIGIN = 0x00000800, LENGTH = 0x00000400
  m_ncache              (RW)  : ORIGIN = 0x20000000, LENGTH = NCACHE_SIZE
  m_data                (RW)  : ORIGIN = 0x20000000 + NCACHE_SIZE, LENGTH = 0x00040000 - NCACHE_SIZE
  m_text                (RX)  : ORIGIN = 0x20240000, LENGTH = TEXT_SIZE
  m_ocram               (RX)  : ORIGIN = 0x20240000 + TEXT_SIZE, LENGTH = 0x00080000 - TEXT_SIZE
  rpmsg_sh_mem          (RW)  : ORIGIN = 0x202C0000, LENGTH = RPMSG_SHMEM_SIZE
  m_heap                (RW)  : ORIGIN = 0x80000000, LENGTH = HEAP_SIZE
  m_sdram               (RX)  : ORIGIN = 0x80000000 + HEAP_SIZE, LENGTH = 0x04000000 - HEAP_SIZE
}

/* Define output sections */
SECTIONS
{
  __NCACHE_REGION_START = ORIGIN(m_ncache);
  __NCACHE_REGION_SIZE  = LENGTH(m_ncache);

  __RPMSG_SH_MEM_START = ORIGIN(rpmsg_sh_mem);
  __RPMSG_SH_MEM_SIZE  = LENGTH(rpmsg_sh_mem);

  /* NOINIT section for rpmsg_sh_mem */
  .noinit_rpmsg_sh_mem (NOLOAD) : ALIGN(4)
  {
     __RPMSG_SH_MEM_START__ = .;
     *(.noinit.$rpmsg_sh_mem*)
     . = ALIGN(4) ;
     __RPMSG_SH_MEM_END__ = .;
  } > rpmsg_sh_mem

  /* section for storing the secondary core image */
  .core1_code :
  {
     . = ALIGN(4) ;
    KEEP (*(.core1_code))
     *(.core1_code*)
     . = ALIGN(4) ;
  } > m_sdram

  /* The startup code goes first into internal RAM */
  .interrupts :
  {
    __VECTOR_TABLE = .;
    __Vectors = .;
    . = ALIGN(4);
    KEEP(*(.isr_vector))     /* Startup code */
    . = ALIGN(4);
  } > m_interrupts

  .a71ch :
  {
     . = ALIGN(4);
     __a71ch_start = .;
     *liblibs_a71ch.a:*(.text .text*)
     *liblibs_a71ch.a:*(.data .data*)
     *liblibs_a71ch.a:*(.rodata .rodata*)
     __a71ch_end = .;
     . = ALIGN(4);
  } > m_ocram

  .curl :
  {
     . = ALIGN(4);
     __curl_start = .;
     *liblibs_curl.a:*(.text .text*)
     *liblibs_curl.a:*(.data .data*)
     *liblibs_curl.a:*(.rodata .rodata*)
     __curl_end = .;
     . = ALIGN(4);
  } > m_ocram

  .mbedtls :
  {
     . = ALIGN(4);
     __mbedtls_start = .;
     *liblibs_nxp_rt1176-sdk-mbedtls.a:*(.text .text*)
     *liblibs_nxp_rt1176-sdk-mbedtls.a:*(.data .data*)
     *liblibs_nxp_rt1176-sdk-mbedtls.a:*(.rodata .rodata*)
     __mbedtls_end = .;
     . = ALIGN(4);
  } > m_ocram

  .lwip :
  {
     . = ALIGN(4);
     __lwip_start = .;
     *liblibs_nxp_rt1176-sdk_lwip.a:*(.text .text*)
     *liblibs_nxp_rt1176-sdk_lwip.a:*(.data .data*)
     *liblibs_nxp_rt1176-sdk_lwip.a:*(.rodata .rodata*)
     *liblibs_nxp_rt1176-sdk_lwip_httpd.a:*(.text .text*)
     *liblibs_nxp_rt1176-sdk_lwip_httpd.a:*(.data .data*)
     *liblibs_nxp_rt1176-sdk_lwip_httpd.a:*(.rodata .rodata*)
     *liblibs_nxp_rt1176-sdk_lwip_mdns.a:*(.text .text*)
     *liblibs_nxp_rt1176-sdk_lwip_mdns.a:*(.data .data*)
     *liblibs_nxp_rt1176-sdk_lwip_mdns.a:*(.rodata .rodata*)
     __lwip_end = .;
     . = ALIGN(4);
  } > m_ocram

  .wiced :
  {
     . = ALIGN(4);
     __wiced_start = .;
     *liblibs_nxp_rt1176-sdk_wiced.a:*(.text .text*)
     *liblibs_nxp_rt1176-sdk_wiced.a:*(.data .data*)
     *liblibs_nxp_rt1176-sdk_wiced.a:*(.rodata .rodata*)
     __wiced_end = .;
     . = ALIGN(4);
  } > m_ocram

  .tensorflow :
  {
     . = ALIGN(4);
     __tensorflow_start = .;
     *liblibs_tensorflow-m7.a:*(.text .text*)
     *liblibs_tensorflow-m7.a:*(.data .data*)
     *liblibs_tensorflow-m7.a:*(.rodata .rodata*)
     *liblibs_kissfft-m7.a*:*(.text .text*)
     *liblibs_kissfft-m7.a*:*(.data .data*)
     *liblibs_kissfft-m7.a*:*(.rodata .rodata*)
     __tensorflow_end = .;
     . = ALIGN(4);
  } > m_ocram

  .arduino :
  {
     . = ALIGN(4);
     __arduino_start = .;
     *liblibs_arduino_coral_micro_bundled.a:*(.text .text*)
     *liblibs_arduino_coral_micro_bundled.a:*(.data .data*)
     *liblibs_arduino_coral_micro_bundled.a:*(.rodata .rodata*)
     *liblibs_arduino_coral_micro_poe_bundled.a:*(.text .text*)
     *liblibs_arduino_coral_micro_poe_bundled.a:*(.data .data*)
     *liblibs_arduino_coral_micro_poe_bundled.a:*(.rodata .rodata*)
     *liblibs_arduino_coral_micro_wifi_bundled.a:*(.text .text*)
     *liblibs_arduino_coral_micro_wifi_bundled.a:*(.data .data*)
     *liblibs_arduino_coral_micro_wifi_bundled.a:*(.rodata .rodata*)
     __arduino_end = .;
     . = ALIGN(4);
  } > m_ocram

  .libjpeg :
  {
     . = ALIGN(4);
     __libjpeg_start = .;
     *liblibs_libjpeg.a:*(.text .text*)
     *liblibs_libjpeg.a:*(.data .data*)
     *liblibs_libjpeg.a:*(.rodata .rodata*)
     __libjpeg_end = .;
     . = ALIGN(4);
  } > m_ocram

  .edgefast_bluetooth_text :
  {
    . = ALIGN(4);
    __edgefast_bluetooth_text_start__ = .;
    *libedgefast_bluetooth_internal.a:*(.text .text*)
    *lib_crypto_m7.a:*(.text .text*)
    *libethermind_ble_protocol.a:*(.text .text*)
    *libethermind_ble_gatt.a:*(.text .text*)
    *libethermind_ble_core.a:*(.text .text*)
    *libethermind_ble_util.a:*(.text .text*)
    __edgefast_bluetooth_text_end__ = .;
    . = ALIGN(4);
  } > m_sdram

  .edgefast_bluetooth_data :
  {
    . = ALIGN(4);
    __edgefast_bluetooth_data_start__ = .;
    *libedgefast_bluetooth_internal.a:*(.data .data*)
    *lib_crypto_m7.a:*(.data .data*)
    *libethermind_ble_protocol.a:*(.data .data*)
    *libethermind_ble_gatt.a:*(.data .data*)
    *libethermind_ble_core.a:*(.data .data*)
    *libethermind_ble_util.a:*(.data .data*)
    __edgefast_bluetooth_data_end__ = .;
    . = ALIGN(4);
  } > m_sdram

  .edgefast_bluetooth_rodata :
  {
    . = ALIGN(4);
    __edgefast_bluetooth_rodata_start__ = .;
    *libedgefast_bluetooth_internal.a:*(.rodata .rodata*)
    *lib_crypto_m7.a:*(.rodata .rodata*)
    *libethermind_ble_protocol.a:*(.rodata .rodata*)
    *libethermind_ble_gatt.a:*(.rodata .rodata*)
    *libethermind_ble_core.a:*(.rodata .rodata*)
    *libethermind_ble_util.a:*(.rodata .rodata*)
    __edgefast_bluetooth_rodata_end__ = .;
    . = ALIGN(4);
  } > m_sdram

  /* The program code and other data goes into internal RAM */
  .text :
  {
    . = ALIGN(4);
    *(.text)                 /* .text sections (code) */
    *(.text*)                /* .text* sections (code) */
    *(.rodata)               /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)              /* .rodata* sections (constants, strings, etc.) */
    KEEP(*(.rodata.debug))
    *(.glue_7)               /* glue arm to thumb code */
    *(.glue_7t)              /* glue thumb to arm code */
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
  } > m_text

  ._settings_handler_static :
  {
    . = ALIGN(4);
    _settings_handler_static_list_start = .;
    KEEP(*(SORT(._settings_handler_static.static.*)))
    _settings_handler_static_list_end = .;
    . = ALIGN(4);
  } > m_text

  ._bt_gatt_service_static :
  {
    . = ALIGN(4);
    _bt_gatt_service_static_list_start = .;
    KEEP(*(SORT(._bt_gatt_service_static.static.*)))
    _bt_gatt_service_static_list_end = .;
    . = ALIGN(4);
  } > m_text

  ._bt_l2cap_fixed_chan :
  {
    . = ALIGN(4);
    _bt_l2cap_fixed_chan_list_start = .;
    KEEP(*(SORT(._bt_l2cap_fixed_chan.static.*)))
    _bt_l2cap_fixed_chan_list_end = .;
    . = ALIGN(4);
  } > m_text

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > m_text

  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } > m_text

 .ctors :
  {
    __CTOR_LIST__ = .;
    /* gcc uses crtbegin.o to find the start of
       the constructors, so we make sure it is
       first.  Because this is a wildcard, it
       doesn't matter if the user does not
       actually link against crtbegin.o; the
       linker won't look for a file to match a
       wildcard.  The wildcard also means that it
       doesn't matter which directory crtbegin.o
       is in.  */
    KEEP (*crtbegin.o(.ctors))
    KEEP (*crtbegin?.o(.ctors))
    /* We don't want to include the .ctor section from
       from the crtend.o file until after the sorted ctors.
       The .ctor section from the crtend file contains the
       end of ctors marker and it must be last */
    KEEP (*(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors))
    KEEP (*(SORT(.ctors.*)))
    KEEP (*(.ctors))
    __CTOR_END__ = .;
  } > m_text

  .dtors :
  {
    __DTOR_LIST__ = .;
    KEEP (*crtbegin.o(.dtors))
    KEEP (*crtbegin?.o(.dtors))
    KEEP (*(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors))
    KEEP (*(SORT(.dtors.*)))
    KEEP (*(.dtors))
    __DTOR_END__ = .;
  } > m_text

  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > m_text

  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > m_text

  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > m_text

  __etext = .;    /* define a global symbol at end of code */
  __DATA_ROM = .; /* Symbol is used by startup for data initialization */

  __VECTOR_RAM = ORIGIN(m_interrupts);
  __RAM_VECTOR_TABLE_SIZE_BYTES = 0x0;

  .data : AT(__DATA_ROM)
  {
    . = ALIGN(4);
    __DATA_RAM = .;
    __data_start__ = .;      /* create a global symbol at data start */
    *(m_usb_dma_init_data)
    *(.data)                 /* .data sections */
    *(.data*)                /* .data* sections */
    *(.got)
    *(.got*)
    *(.got.plt)
    *(.got.plt*)
    KEEP(*(.jcr*))
    . = ALIGN(4);
    _net_buf_pool_list = .;
    KEEP(*(SORT(._net_buf_pool*)))
    . = ALIGN(4);
    __data_end__ = .;        /* define a global symbol at data end */
  } > m_data
  __NDATA_ROM = __DATA_ROM + (__data_end__ - __data_start__);
  .ncache.init : AT(__NDATA_ROM)
  {
    __noncachedata_start__ = .;   /* create a global symbol at ncache data start */
    *(NonCacheable.init)
    . = ALIGN(4);
    __noncachedata_init_end__ = .;   /* create a global symbol at initialized ncache data end */
  } > m_ncache
  . = __noncachedata_init_end__;
  .ncache :
  {
    *(NonCacheable)
    . = ALIGN(4);
    __noncachedata_end__ = .;     /* define a global symbol at ncache data end */
  } > m_ncache
  __SDRAM_ROM = __NDATA_ROM + (__noncachedata_init_end__ - __noncachedata_start__);
  .sdram_data : AT(__SDRAM_ROM)
  {
    . = ALIGN(4);
    __sdram_data_start__ = .;
    KEEP(*(.sdram_data*))
    . = ALIGN(4);
    __sdram_data_end__ = .;
  } > m_sdram

  /* Uninitialized data section */
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss section */
    . = ALIGN(4);
    __START_BSS = .;
    __bss_start__ = .;
    *(m_usb_dma_noninit_data)
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
    __END_BSS = .;
  } > m_data
  __DATA_END = __SDRAM_ROM;
  text_end = ORIGIN(m_text) + LENGTH(m_text);
  ASSERT(__DATA_END <= text_end, "region m_text overflowed with text and data")

  .sdram_bss (NOLOAD) :
  {
    . = ALIGN(4);
    __sdram_bss_start__ = .;
    *(.sdram_bss*)
    /* edgefast_bluetooth_bss */
    *lib_crypto_m7.a:*(.bss .bss*)
    *libedgefast_bluetooth_internal.a:*(.bss .bss*)
    *libethermind_ble_protocol.a:*(.bss .bss*)
    *libethermind_ble_gatt.a:*(.bss .bss*)
    *libethermind_ble_core.a:*(.bss .bss*)
    *libethermind_ble_util.a:*(.bss .bss*)
    . = ALIGN(4);
    __sdram_bss_end__ = .;
  } > m_sdram

  .heap :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    __HeapBase = .;
    . += HEAP_SIZE;
    __HeapLimit = .;
    __heap_limit = .; /* Add for _sbrk */
  } > m_heap

  .stack :
  {
    . = ALIGN(8);
    . += STACK_SIZE;
  } > m_data

  /* Initializes stack on the end of block */
  __StackTop   = ORIGIN(m_data) + LENGTH(m_data);
  __StackLimit = __StackTop - STACK_SIZE;
  PROVIDE(__stack = __StackTop);

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
