"""Local evaluation harness.

The competition allows only 5 submissions/day, so the iterate -> measure loop
must happen locally. Nothing should reach the ladder without passing through
``evaluate`` first. Everything here requires the Linux engine, so run it inside
the Docker image (see Makefile) or on a Linux box.
"""
