import pygame as pg
import image_main as i_main
import os


def run_app():
    pg.init()

    # Create canvas
    width = 1280
    height = 720
    canvas = pg.display.set_mode((width, height))

    # Title of canvas
    pg.display.set_caption("IMAGE COMPRESSOR/BUILDER")
    exit = False

    # Background colour
    canvas.fill(color=(20, 20, 22))

    # Set the font
    font = pg.font.Font(None, 28)

    # Make the buttons for compression
    s_compress_button = pg.Rect(width / 2 - 250 / 2, 50, 250, 50)
    text = "Simple compression"  # Text to display within the button
    text_s_comp = font.render(text, True, (0, 0, 0))
    text_s_comp_rect = text_s_comp.get_rect(center=s_compress_button.center)  # Center the text within the button

    s_build_button = pg.Rect(width / 2 - 250 / 2, 150, 250, 50)
    text = "Simple build"  # Text to display within the button
    text_s_build = font.render(text, True, (0, 0, 0))
    text_s_build_rect = text_s_build.get_rect(center=s_build_button.center)  # Center the text within the button

    c_compress_button = pg.Rect(width / 2 - 250 / 2, 300, 250, 50)
    text = "Complex compression"  # Text to display within the button
    text_c_comp = font.render(text, True, (0, 0, 0))
    text_c_comp_rect = text_c_comp.get_rect(center=c_compress_button.center)  # Center the text within the button

    c_build_button = pg.Rect(width / 2 - 250 / 2, 400, 250, 50)
    text = "Complex build"  # Text to display within the button
    text_c_build = font.render(text, True, (0, 0, 0))
    text_c_build_rect = text_c_build.get_rect(center=c_build_button.center)  # Center the text within the button

    # Make the base rectangles for the images
    img_in = pg.Rect(50, 50, 400, 400)
    selected_in_img = ""
    update_in_image = False

    img_out = pg.Rect(width - 450, 50, 400, 400)
    selected_out_img = ""
    update_out_image = False

    # Draw the images on the canvas
    pg.draw.rect(canvas, (150, 150, 150), img_in)
    pg.draw.rect(canvas, (150, 150, 150), img_out)

    # Start running
    while not exit:
        # Check for the input image files
        input_files = [f for f in os.listdir("images/IN/") if f.endswith(".png")]

        # Check for the data files
        data_files = [f for f in os.listdir("images/STORE/") if f.endswith(".pickle")]

        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit = True
            if event.type == pg.MOUSEBUTTONDOWN:
                # Check which button was pressed
                if s_compress_button.collidepoint(event.pos):
                    if selected_in_img != "":
                        print("Starting simple compression")
                        print(selected_in_img)
                        i_main.compress_alg_s(selected_in_img)

                if s_build_button.collidepoint(event.pos):
                    # Only build the data if it was produced with the right algorithm
                    if selected_out_img != "" and selected_out_img[
                                                  len(selected_out_img) - len(".pickle") - len("simp"):len(
                                                          selected_out_img) - len(".pickle")] == "simp":
                        print("Starting simple building")
                        print(selected_out_img)
                        i_main.build_alg_s(selected_out_img)
                        update_out_image = True

                if c_compress_button.collidepoint(event.pos):
                    if selected_in_img != "":
                        print("Starting complex compression")
                        print(selected_in_img)
                        i_main.compress_alg_c(selected_in_img)

                if c_build_button.collidepoint(event.pos):
                    # Only build the data if it was produced with the right algorithm
                    if selected_out_img != "" and selected_out_img[
                                                  len(selected_out_img) - len(".pickle") - len("simp"):len(
                                                          selected_out_img) - len(".pickle")] == "comp":
                        print("Starting complex building")
                        print(selected_out_img)
                        i_main.build_alg_c(selected_out_img)
                        update_out_image = True

                # Check if a click occurred on an input file name
                for i, file in enumerate(input_files):
                    text_rect = font.render(file, True, (0, 0, 0), (200, 200, 200)).get_rect(
                        topleft=(50, 500 + i * 30))
                    if text_rect.collidepoint(event.pos):
                        print("Clicked on input file:", file)
                        selected_in_img = file
                        update_in_image = True

                        # Display the selected input file

                # Check if a click occurred on an output file name
                for i, file in enumerate(data_files):
                    text_rect = font.render(file, True, (0, 0, 0), (200, 200, 200)).get_rect(
                        topleft=(width - 450, 500 + i * 30))
                    if text_rect.collidepoint(event.pos):
                        print("Clicked on output file:", file)
                        selected_out_img = file
                        # Perform further actions for the selected file

        # Draw the buttons on the canvas
        pg.draw.rect(canvas, (200, 200, 200), s_compress_button)
        canvas.blit(text_s_comp, text_s_comp_rect)
        pg.draw.rect(canvas, (200, 200, 200), s_build_button)
        canvas.blit(text_s_build, text_s_build_rect)
        pg.draw.rect(canvas, (200, 200, 200), c_compress_button)
        canvas.blit(text_c_comp, text_c_comp_rect)
        pg.draw.rect(canvas, (200, 200, 200), c_build_button)
        canvas.blit(text_c_build, text_c_build_rect)

        # Draw the input file names
        text_y = 500  # Initial y-coordinate for text
        for file in input_files:
            text = font.render(file, True, (0, 0, 0), (200, 200, 200))
            canvas.blit(text, (50, text_y))
            text_y += 30

        # Draw the output file names
        text_y = 500  # Initial y-coordinate for text
        for file in data_files:
            text = font.render(file, True, (0, 0, 0), (200, 200, 200))
            canvas.blit(text, (width - 450, text_y))
            text_y += 30

        if update_in_image:
            in_image = pg.image.load("images/IN/" + selected_in_img)
            resized_image = pg.transform.scale(in_image, (img_in.width, img_in.height))
            canvas.blit(resized_image, (img_in.x, img_in.y))
            update_in_image = False

        if update_out_image:
            print("Updating out image")
            image_name = selected_out_img[:len(selected_out_img) - len(".pickle")]
            out_image = pg.image.load("images/OUT/" + image_name + ".png")
            resized_image = pg.transform.scale(out_image, (img_out.width, img_out.height))
            canvas.blit(resized_image, (img_out.x, img_out.y))
            update_out_image = False

        pg.display.update()


if __name__ == "__main__":
    run_app()
