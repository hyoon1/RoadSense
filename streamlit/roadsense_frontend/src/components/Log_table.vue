<script setup lang="ts">
import { ref } from 'vue'
const products = ref(null)

var url =
  'http://localhost:5000/roadcondition/notifications?APIKey=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c&road_section_id=wtl_a1234'

fetch(url)
  .then((response) => response.json())
  .then((data) => {
    products.value = data.data
  })
  .catch(() => {
    products.value = []
  })
</script>

<template>
  <section class="ftco-section">
    <div class="container">
      <div class="row justify-content-center">
        <div class="col-md-6 text-center mb-5">
          <h2 class="heading-section">Records of poor conditions</h2>
        </div>
      </div>
      <div class="row">
        <div class="col-md-12">
          <div class="table-wrap">
            <table class="table table-bordered table-dark table-hover">
              <thead>
                <tr>
                  <th>#</th>
                  <th>damage_type</th>
                  <th>latitude</th>
                  <th>longitude</th>
                  <th>severity</th>
                  <th>reported_at</th>
                </tr>
              </thead>
              <tbody v-if="products">
                <tr v-for="data in products" :key="data.id">
                  <th scope="row">{{ data.id }}</th>
                  <td>{{ data.damage_type }}</td>
                  <td>{{ data.latitude }}</td>
                  <td>{{ data.longitude }}</td>
                  <td>{{ data.severity }}</td>
                  <td>{{ data.reported_at }}</td>
                </tr>
                <!--
                <tr>
                  <th scope="row">1</th>
                  <td>1</td>
                  <td>1</td>
                  <td>1</td>
                  <td>1</td>
                  <td>1</td>
                </tr>-->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
    <div v-if="!products" class="text-center">
      <div class="spinner-border spinner-border-sm"></div>
    </div>
  </section>
</template>
