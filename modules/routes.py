from app import app
from flask import Blueprint, render_template, session

bp = Blueprint('routes', __name__)


@bp.route('/about')
def home():
    return render_template('about.html', current_route="about")

@bp.route('/contact')
def contact():
    return render_template('contact.html', current_route="contact")

@bp.route('/add-user')
def register():
    return render_template('adduser.html', current_route="add-user")

@bp.route('/vote')
def vote():
    return render_template('vote.html', current_route="vote")