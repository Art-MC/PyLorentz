import PySimpleGUI as sg


# ---------  Window Scaling  --------- #
def window_scaling():
    """Properly set the scaling to look
    similar across OS."""

    sys_layout = [[sg.Text('')]]
    scaling_window = sg.Window('Window Title', sys_layout, no_titlebar=True, alpha_channel=0).Finalize()
    scaling_window.TKroot.tk.call('tk', 'scaling', 1)
    scaling_window.close()


def pad(left, right, top, bottom):
    """Set padding of element.

    Parameters
    ----------
    left : int
        Left padding.
    right : int
        Right padding.
    top : int
        Top padding.
    bottom : int
        Bottom padding.

    Returns
    ----------
    padding : tuple of tuples of ints
        The padding format for PySimpleGUI/Tkinter
    """

    padding = ((left, right), (top, bottom))
    return padding


# ---------  Window & Element Defaults --------- #
class WindowStyle(object):
    """The WindowStyle class sets the styling
    of the window and all elements."""

    def __init__(self):
        """Initialize the style of the window."""

        # ---------  Theme & Scaling  --------- #
        window_scaling()
        self.theme = sg.theme('BlueMono')
        self.DEF_BACKGROUND = sg.theme_background_color()

        # ---------  self.Fonts and test  --------- #
        self.fonts = {
            'title': ('Times New Roman', '24', 'bold'),
            'bold heading': ('Times New Roman', '18', 'bold'),
            'body': ('Times New Roman', '16'),
            'heading': ('Times New Roman', '18'),
            'tab': ('Times New Roman', '16')}
        self.DEF_FONT = self.fonts['body']
        self.options = sg.SetOptions(margins=(0, 0), element_padding=((0, 0), (0, 0)),
                                     border_width=0, font=self.DEF_FONT)
        self.window_width = 1170
        self.window_height = 800
        self.tab_size = (self.window_width, self.window_height - 25)
        self.small_tab_size = (self.window_width, self.window_height - 50)

    # ---------  Individual Element Styling --------- #
    def styles(self, key):
        """The styles within the GUI.

        Parameters
        ----------
        key : str
            The key for a PySimpleGUI element.

        Returns
        -------
        key_style : dict
            A dictionary of the key-value pairs
            for a specific type of element given
            by 'key'."""

        # Input size
        inp_pad = ((0, 0), (0, 10))
        inp_wid = (5, 1)
        inp_d = {'pad': inp_pad,
                 'size': inp_wid}

        # Fonts
        T_f = self.fonts["title"]
        bH_f = self.fonts["bold heading"]
        # B_f = self.fonts['body']
        # H_f = self.fonts['heading']
        # Tab_f = self.fonts['tab']

        def home_style(key, val):
            # __________________ Home  ______________ #
            # Browse Button elements
            if key == '__Fiji_Browse__':
                val.update(pad=pad(164, 0, 10, 0), target='__Fiji_Path__')
            # Button elements
            elif key == '__Fiji_Set__':
                val.update(pad=pad(10, 0, 10, 0))
            elif key == '__Fiji_Reset__':
                val.update(pad=pad(10, 0, 10, 0))
            # Input elements
            elif key == '__Fiji_Path__':
                val.update(pad=pad(16, 0, 10, 0), size=(30, 1), enable_events=True,
                           metadata={'Set': '', 'Def': '', 'State': 'Def'})
            # Text elements
            elif key == 'home_title':
                val.update(pad=pad(246, 246, 30, 5), font=T_f)
            elif key == 'home_version':
                val.update(pad=(346, 0), font=bH_f)
            elif key == 'home_authors':
                val.update(pad=(239, 0))
            elif key == 'home_readme':
                val.update(pad=(225, 5), justification='l')
            elif key == 'home_contact':
                val.update(pad=pad(255, 0, 20, 40))
            return val

        def LS_style(key, val):
            # _______________ Linear Sift Tab _______________ #
            # Invisible
            if key == '__LS_Staging_Load__':
                val.update(visible=False, enable_events=True)
            # Browse buttons element
            elif key == '__LS_Image_Dir_Browse__':
                val.update(pad=pad(8, 0, 0, 0), target='__LS_Image_Dir_Path__')
            elif key == '__LS_Load_Stack__':
                val.update(pad=pad(130, 0, 0, 0), target='__LS_Staging_Load__')
            # Button element
            elif key == '__LS_Set_Img_Dir__':
                val.update(pad=pad(5, 0, 0, 0))
            elif key == '__LS_Reset_Img_Dir__':
                val.update(pad=pad(5, 0, 0, 0))
            elif key == '__LS_Run_Align__':
                val.update(pad=pad(55, 0, 5, 0), disabled=True)
            elif key == '__LS_Adjust__':
                val.update(pad=pad(20, 0, 0, 0), disabled=True,
                           metadata={'Set': 'Quit Adjustment', 'Def': 'Adjust Image',
                                     'State': 'Def'})
            elif key == '__LS_View_Stack__':
                val.update(pad=pad(5, 0, 0, 0), disabled=True,
                           metadata={'Set': 'Hide Stack', 'Def': 'View Stack',
                                     'State': 'Def'})
            # Checkbox element
            elif key == '__LS_horizontal_flip__':
                val.update(default=True, pad=pad(35, 0, 0, 0))
            elif key == '__LS_interp__':
                val.update(pad=pad(200, 0, 4, 20), default=True)
            # Combo element
            elif key == '__LS_exp_transf__':
                val.update(pad=pad(0, 10, 4, 10), size=(9, 1),
                           default_value='Affine', readonly=True)
            # Graph element
            elif key == '__LS_Graph__':
                val.update(background_color='#808080', pad=pad(14, 14, 10, 0))
            # Input field element
            elif key == '__LS_Image_Dir_Path__':
                val.update(size=(30, 1))
            elif key == '__LS_Image1__':
                default_text = 'None'
                val.update(default_text=default_text, pad=pad(0, 0, 0, 0),
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__LS_Image2__':
                val.update(default_text='None', pad=pad(0, 0, 0, 0),
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__LS_Stack__':
                val.update(default_text='None', enable_events=True, pad=pad(19, 0, 0, 5),
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__LS_igb__':
                val.update(**inp_d)
            elif key == '__LS_spso__':
                val.update(**inp_d)
            elif key == '__LS_min_im__':
                val.update(**inp_d)
            elif key == '__LS_max_im__':
                val.update(**inp_d)
            elif key == '__LS_fds__':
                val.update(**inp_d)
            elif key == '__LS_fdob__':
                val.update(**inp_d)
            elif key == '__LS_cncr__':
                val.update(**inp_d)
            elif key == '__LS_max_al_err__':
                val.update(**inp_d)
            elif key == '__LS_inlier_rat__':
                val.update(**inp_d)
            elif key == '__LS_transform_x__':
                val.update(pad=(5, 5), size=(4, 1))
            elif key == '__LS_transform_y__':
                val.update(pad=(5, 5), size=(4, 1))
            elif key == '__LS_transform_rot__':
                val.update(pad=(5, 5), size=(4, 1))
            # Output element
            elif key == "__LS_Log__":
                val.update(size=(40, 10), pad=pad(10, 0, 5, 5), background_color="light grey")
            # Radio element
            elif key == '__LS_full_align__':
                val.update(default=True, size=(12, 2))
            elif key == '__LS_param_test__':
                val.update(size=(22, 2))
            elif key == '__LS_unflip_reference__':
                val.update(default=True, size=(12, 0), enable_events=True)
            elif key == '__LS_flip_reference__':
                val.update(size=(12, 0), enable_events=True)
            # Slider element
            elif key == '__LS_Stack_Slider__':
                val.update(range=(0, 2), size=(27, 15), pad=pad(0, 0, 2, 4), default_value=1,
                           enable_events=True, resolution=1, orientation='h', disable_number_display=True,
                           visible=False,
                           metadata={'slider_range': (0, 2)})
            return val

        def BUJ_style(key, val):
            # _______________ bUnwarpJ Tab _______________ #
            # Invisible
            if key == '__BUJ_Unflip_Stage_Load__':
                val.update(visible=False, enable_events=True)
            if key == '__BUJ_Flip_Stage_Load__':
                val.update(visible=False, enable_events=True)
            if key == '__BUJ_Stack_Stage_Load__':
                val.update(visible=False, enable_events=True)
            if key == '__BUJ_Mask_Stage_Load__':
                val.update(visible=False, enable_events=True)
            # Browse buttons element
            elif key == '__BUJ_Image_Dir_Browse__':
                val.update(pad=pad(8, 0, 0, 0), target='__BUJ_Image_Dir_Path__')
            elif key == '__BUJ_Load_Flip_Stack__':
                val.update(pad=pad(0, 0, 5, 0), target='__BUJ_Flip_Stage_Load__')
            elif key == '__BUJ_Load_Unflip_Stack__':
                val.update(pad=pad(0, 0, 0, 0), target='__BUJ_Unflip_Stage_Load__')
            elif key == '__BUJ_Load_Stack__':
                val.update(pad=pad(100, 0, 10, 10), target='__BUJ_Stack_Stage_Load__')

            elif key == '__BUJ_Load_Mask__':
                val.update(pad=pad(5, 0, 0, 0), target='__BUJ_Mask_Stage_Load__',
                           enable_events=True)

            # Button element
            elif key == '__BUJ_Set_Img_Dir__':
                val.update(pad=pad(5, 0, 0, 0))
            elif key == '__BUJ_Reset_Img_Dir__':
                val.update(pad=pad(5, 0, 0, 0))
            elif key == '__BUJ_Flip_Align__':
                val.update(pad=pad(0, 0, 5, 0), disabled=True)
            elif key == '__BUJ_Unflip_Align__':
                val.update(pad=pad(0, 0, 0, 0), disabled=True)
            elif key == '__BUJ_Adjust__':
                val.update(pad=pad(45, 0, 0, 0), disabled=True,
                           metadata={'Set': 'Quit Adjustment', 'Def': 'Adjust Image',
                                     'State': 'Def'})
            elif key == '__BUJ_Make_Mask__':
                val.update(pad=pad(30, 5, 0, 0), disabled=True,
                           metadata={'Set': 'Finish Mask', 'Def': 'Create New',
                                     'State': 'Def'})
            elif key == '__BUJ_Reset_Mask__':
                val.update(visible=False)
            elif key == '__BUJ_Elastic_Align__':
                val.update(pad=pad(38, 0, 3, 0), disabled=True)
            elif key == '__BUJ_View_Stack__':
                val.update(pad=pad(5, 0, 10, 10), disabled=True,
                           metadata={'Set': 'Hide Stack', 'Def': 'View Stack',
                                     'State': 'Def'})
            # Checkbox element
            elif key == '__BUJ_filter__':
                val.update(pad=pad(5, 0, 5, 5), default=True)
            elif key == '__BUJ_LS_interp__':
                val.update(pad=pad(200, 0, 4, 20), default=True)
            elif key == '__BUJ_horizontal_flip__':
                val.update(default=True,  pad=pad(45, 0, 0, 0))
            # Combo element
            elif key == '__BUJ_exp_transf__':
                val.update(pad=pad(0, 10, 5, 10), size=(9, 1),
                           default_value='Affine', readonly=True)
            elif key == '__BUJ_LS_exp_transf__':
                val.update(pad=pad(0, 10, 4, 10), size=(9, 1),
                           default_value='Affine', readonly=True)
            elif key == '__BUJ_reg_mode__':
                val.update(pad=pad(0, 10, 19, 5), size=(11, 1),
                           default_value='Accurate', enable_events=True, readonly=True)
            elif key == '__BUJ_init_def__':
                val.update(pad=pad(0, 10, 4, 5), size=(11, 1),
                           default_value='Very Coarse', readonly=True)
            elif key == '__BUJ_final_def__':
                val.update(pad=pad(0, 10, 4, 5), size=(11, 1),
                           default_value='Coarse', readonly=True)
            elif key == '__BUJ_Mask_View__':
                val.update(pad=pad(5, 0, 5, 0), visible=True,
                           default_value='Unflip', enable_events=True, readonly=True)
            elif key == '__BUJ_Stack_Choice__':
                val.update(pad=pad(5, 0, 10, 10), visible=True, default_value='Unflip LS', enable_events=True, readonly=True)
            # Graph element
            elif key == '__BUJ_Graph__':
                val.update(background_color='#808080', pad=pad(14, 14, 10, 0),
                           enable_events=True)
            # Input field element
            elif key == '__BUJ_Image_Dir_Path__':
                val.update(size=(30, 1), enable_events=True)
            elif key == '__BUJ_LS_igb__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_spso__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_min_im__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_max_im__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_fds__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_fdob__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_cncr__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_max_al_err__':
                val.update(**inp_d)
            elif key == '__BUJ_LS_inlier_rat__':
                val.update(**inp_d)
            elif key == '__BUJ_igb__':
                val.update(**inp_d)
            elif key == '__BUJ_spso__':
                val.update(**inp_d)
            elif key == '__BUJ_min_im__':
                val.update(**inp_d)
            elif key == '__BUJ_max_im__':
                val.update(**inp_d)
            elif key == '__BUJ_fds__':
                val.update(**inp_d)
            elif key == '__BUJ_fdob__':
                val.update(**inp_d)
            elif key == '__BUJ_cncr__':
                val.update(**inp_d)
            elif key == '__BUJ_max_al_err__':
                val.update(**inp_d)
            elif key == '__BUJ_inlier_rat__':
                val.update(**inp_d)
            elif key == '__BUJ_min_num_inlier__':
                val.update(**inp_d)
            elif key == '__BUJ_div_w__':
                val.update(**inp_d)
            elif key == '__BUJ_curl_w__':
                val.update(**inp_d)
            elif key == '__BUJ_land_w__':
                val.update(**inp_d)
            elif key == '__BUJ_img_w__':
                val.update(**inp_d)
            elif key == '__BUJ_cons_w__':
                val.update(**inp_d)
            elif key == '__BUJ_stop_thresh__':
                val.update(**inp_d)
            elif key == '__BUJ_transform_x__':
                val.update(pad=(5, 5), size=(4, 1))
            elif key == '__BUJ_transform_y__':
                val.update(pad=(5, 5), size=(4, 1))
            elif key == '__BUJ_transform_rot__':
                val.update(pad=(5, 5), size=(4, 1))

            elif key == "__BUJ_Unflip_Mask_Inp__":
                val.update(default_text='None', pad=((0, 0), (5, 0)), size=(18, 1),
                           disabled=True,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == "__BUJ_Flip_Mask_Inp__":
                val.update(default_text='None', pad=((0, 0), (3, 0)), size=(18, 1),
                           disabled=True,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})

            elif key == '__BUJ_Unflip_Stack_Inp__':
                val.update(default_text='None', size=(30, 1),
                           background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__BUJ_Flip_Stack_Inp__':
                val.update(default_text='None', size=(30, 1),
                           background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__BUJ_Image1__':
                val.update(default_text='None',
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__BUJ_Image2__':
                val.update(default_text='None',
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})
            elif key == '__BUJ_Stack__':
                val.update(default_text='None', pad=pad(0, 0, 0, 5), # enable_events=True,
                           size=(30, 1), background_color=self.DEF_BACKGROUND,
                           metadata={'Set': 'None', 'Def': 'None',
                                     'State': 'Def'})

            # metadata = {'Set': 'None', 'Def': 'None',
            #             'State': 'Def'}


            # Mutliline element
            elif key == "__BUJ_Log__":
                val.update(size=(40, 10), autoscroll=True, pad=pad(10, 10, 10, 0),
                           disabled=True, background_color="light grey")
            # Radio element
            elif key == '__BUJ_unflip_reference__':
                val.update(default=True, size=(12, 2), pad=pad(30, 0, 4, 4), enable_events=True)
            elif key == '__BUJ_flip_reference__':
                val.update(size=(22, 2), pad=pad(10, 0, 0, 4), enable_events=True)
            # Slider element
            elif key == '__BUJ_Stack_Slider__':
                val.update(range=(0, 2), size=(32, 15), default_value=1,
                           enable_events=True, resolution=1, orientation='h',
                           disable_number_display=True, visible=False,
                           metadata={'slider_range': (0, 2)})
            elif key == '__BUJ_img_subsf__':
                val.update(range=(0, 7), size=(25, 12), pad=pad(40, 0, 2, 4), default_value=0,
                           resolution=1, orientation='h', disable_number_display=False, visible=True)
            return val

        def REC_style(key, val):
            # Graph
            if key == '__REC_Graph1__':
                val.update(background_color='#808080', pad=pad(14, 14, 10, 0),
                           enable_events=True)
            elif key == '__REC_Graph2__':
                val.update(background_color='#808080', pad=pad(14, 14, 10, 0),
                           enable_events=True)

            return val

        # Check if key a part of tab
        # For correct key, return its style
        key_style = dict(key=key)
        if key.startswith("home") or key.startswith("__Fiji"):
            key_style = home_style(key, key_style)
        elif key.startswith("__LS"):
            key_style = LS_style(key, key_style)
        elif key.startswith("__BUJ"):
            key_style = BUJ_style(key, key_style)
        elif key.startswith("__REC"):
            key_style = REC_style(key, key_style)
        return key_style